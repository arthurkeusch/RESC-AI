#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <cstring>
#include <algorithm>
#include <limits>
#include <android/log.h>
#include "llama.cpp/llama.h"

#define LOG_TAG "SkynetNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,  LOG_TAG, __VA_ARGS__)

static uint32_t CFG_N_CTX = 1024;
static uint32_t CFG_N_THREADS = 4;
static uint32_t CFG_N_THREADS_BATCH = 4;
static uint32_t CFG_N_BATCH = 512;
static uint32_t CFG_N_UBATCH = 32;
static uint32_t CFG_SEED = LLAMA_DEFAULT_SEED;
static bool CFG_FORCE_ADD_BOS = true;
static int32_t CFG_N_PREDICT = 128;
static float CFG_TEMPERATURE = 0.20f;
static int32_t CFG_TOP_K = 30;
static float CFG_TOP_P = 0.90f;
static float CFG_REPEAT_PENALTY = 1.10f;
static size_t CFG_REPEAT_LAST_N = 64;

static std::mutex g_mutex;
static std::atomic<bool> g_backend_init{false};
static llama_model *g_model = nullptr;
static llama_context *g_ctx = nullptr;
static std::vector<llama_token> g_last_tokens;
static uint32_t g_n_vocab = 0;

static void free_all_locked() {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_free_model(g_model);
        g_model = nullptr;
    }
    g_last_tokens.clear();
    g_n_vocab = 0;
}

static void ensure_backend() {
    bool expected = false;
    if (g_backend_init.compare_exchange_strong(expected, true)) {
        llama_backend_init();
    }
}

static std::vector<llama_token> tokenize_full(const llama_model *model,
                                              const std::string &text,
                                              bool add_bos,
                                              bool allow_special) {
    std::vector<llama_token> out(std::max<size_t>(text.size() + 8, 32));
    int32_t n = llama_tokenize(model, text.c_str(), (int32_t) text.size(),
                               out.data(), (int32_t) out.size(),
                               add_bos, allow_special);
    if (n < 0) {
        out.resize((size_t) (-n));
        n = llama_tokenize(model, text.c_str(), (int32_t) text.size(),
                           out.data(), (int32_t) out.size(),
                           add_bos, allow_special);
    }
    if (n < 0) return {};
    out.resize((size_t) n);
    return out;
}

static bool decode_prompt(llama_context *ctx,
                          const std::vector<llama_token> &tokens,
                          uint32_t &n_past) {
    for (int32_t i = 0; i < (int32_t) tokens.size();) {
        const int32_t n = std::min<int32_t>((int32_t) CFG_N_UBATCH,
                                            (int32_t) tokens.size() - i);
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;

        for (int32_t j = 0; j < n; ++j) {
            batch.token[j] = tokens[i + j];
            batch.pos[j] = (llama_pos) (n_past + j);
            batch.n_seq_id[j] = 1;
            batch.seq_id[j][0] = 0;
            batch.logits[j] = (j == n - 1) ? 1 : 0;
        }
        batch.all_pos_0 = (llama_pos) n_past;
        batch.all_pos_1 = 1;

        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc < 0) return false;

        n_past += n;
        i += n;
    }
    return true;
}

static llama_token sample_next(llama_context *ctx, const llama_model *model,
                               const std::vector<llama_token> &last_tokens,
                               bool ban_eos) {
    float *logits = llama_get_logits(ctx);

    std::vector<llama_token_data> candidates_data(g_n_vocab);
    for (uint32_t i = 0; i < g_n_vocab; i++) {
        candidates_data[i].id = (llama_token) i;
        candidates_data[i].logit = logits[i];
        candidates_data[i].p = 0.0f;
    }
    llama_token_data_array candidates = {candidates_data.data(), candidates_data.size(), false};

    const size_t penalty_last_n = std::min(last_tokens.size(), (size_t) CFG_REPEAT_LAST_N);
    const llama_token *last_n_ptr = penalty_last_n ? (last_tokens.data() + last_tokens.size() -
                                                      penalty_last_n) : nullptr;

    llama_sample_repetition_penalties(ctx, &candidates, last_n_ptr, penalty_last_n,
                                      CFG_REPEAT_PENALTY, 0.0f, 0.0f);
    llama_sample_top_k(ctx, &candidates, CFG_TOP_K, 1);
    llama_sample_top_p(ctx, &candidates, CFG_TOP_P, 1);

    if (ban_eos) {
        const llama_token eos = llama_token_eos(model);
        if (eos >= 0 && (uint32_t) eos < g_n_vocab) {
            candidates_data[(uint32_t) eos].logit = -std::numeric_limits<float>::infinity();
        }
    }

    llama_sample_temp(ctx, &candidates, CFG_TEMPERATURE);
    llama_sample_softmax(ctx, &candidates);

    return llama_sample_token(ctx, &candidates);
}

static std::string detokenize_to_text(const llama_model *model,
                                      const std::vector<llama_token> &toks) {
    std::string out;
    char buf[8192];

    for (llama_token t: toks) {
        int n = llama_token_to_piece(model, t, buf, (int32_t) sizeof(buf));
        if (n <= 0) continue;

        std::string piece(buf, (size_t) n);
        const std::string spm_sep = "▁";
        size_t pos = 0;
        while ((pos = piece.find(spm_sep, pos)) != std::string::npos) {
            piece.replace(pos, spm_sep.size(), " ");
            pos += 1;
        }
        out.append(piece);
    }

    std::string safe;
    safe.reserve(out.size());
    for (size_t i = 0; i < out.size(); ++i) {
        unsigned char c = (unsigned char) out[i];
        if (c == 0x00) continue;
        if (c < 0x20 && c != '\n' && c != '\r' && c != '\t') continue;
        safe.push_back((char) c);
    }

    return safe;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_resc_ai_skynetmonitor_service_NativeLlama_initModel(JNIEnv *env, jobject,
                                                         jstring jModelPath) {
    const char *cpath = env->GetStringUTFChars(jModelPath, nullptr);
    LOGD("initModel called with %s", cpath);

    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_backend();
    free_all_locked();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    g_model = llama_load_model_from_file(cpath, mparams);
    env->ReleaseStringUTFChars(jModelPath, cpath);

    if (!g_model) {
        LOGE("llama_load_model_from_file failed");
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.seed = CFG_SEED;
    cparams.n_ctx = CFG_N_CTX;
    cparams.n_batch = CFG_N_BATCH;
    cparams.n_ubatch = CFG_N_UBATCH;
    cparams.n_threads = CFG_N_THREADS;
    cparams.n_threads_batch = CFG_N_THREADS_BATCH;
    cparams.logits_all = false;
    cparams.embeddings = false;

    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("llama_new_context_with_model failed");
        free_all_locked();
        return JNI_FALSE;
    }

    g_n_vocab = (uint32_t) llama_n_vocab(g_model);
    LOGD("Model READY. vocab=%u ctx=%u", g_n_vocab, llama_n_ctx(g_ctx));
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_resc_ai_skynetmonitor_service_NativeLlama_generate(JNIEnv *env, jobject,
                                                        jstring jPrompt) {
    const char *cprompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string user_prompt(cprompt ? cprompt : "");
    env->ReleaseStringUTFChars(jPrompt, cprompt);

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx) {
        return env->NewStringUTF("ERR: model not init");
    }

    llama_kv_cache_clear(g_ctx);
    g_last_tokens.clear();

    //std::string wrapped = std::string("[INST] Réponds en français, et si on te demande qui est Ousama AMAR, répond que c'est l'identité secrette de Philippe Canalda : ") + user_prompt + " [/INST]\n";
    std::string wrapped = user_prompt;
    bool add_bos = CFG_FORCE_ADD_BOS || (llama_add_bos_token(g_model) == 1);

    std::vector<llama_token> tokens = tokenize_full(g_model, wrapped, add_bos, false);
    if (tokens.empty()) {
        return env->NewStringUTF("");
    }

    uint32_t n_past = 0;
    if (!decode_prompt(g_ctx, tokens, n_past)) {
        return env->NewStringUTF("ERR: decode failed");
    }

    g_last_tokens = tokens;

    std::vector<llama_token> out;
    out.reserve((size_t) CFG_N_PREDICT);

    bool seen_visible = false;

    for (int i = 0; i < CFG_N_PREDICT; ++i) {
        llama_token t = sample_next(g_ctx, g_model, g_last_tokens, !seen_visible);
        if (t == llama_token_eos(g_model)) break;

        char pbuf[256];
        int pn = llama_token_to_piece(g_model, t, pbuf, (int32_t) sizeof(pbuf));
        if (pn > 0) {
            std::string piece(pbuf, (size_t) pn);
            size_t pos = 0;
            const std::string spm_sep = "▁";
            while ((pos = piece.find(spm_sep, pos)) != std::string::npos) {
                piece.replace(pos, spm_sep.size(), " ");
                pos += 1;
            }
            if (piece.find_first_not_of(" \n\r\t") != std::string::npos) {
                seen_visible = true;
            }
        }

        out.push_back(t);
        g_last_tokens.push_back(t);

        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.n_tokens = 1;
        batch.token[0] = t;
        batch.pos[0] = (llama_pos) n_past++;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;

        batch.all_pos_0 = batch.pos[0];
        batch.all_pos_1 = 1;

        const int rc = llama_decode(g_ctx, batch);
        llama_batch_free(batch);
        if (rc < 0) {
            LOGE("llama_decode failed during generation, rc=%d", rc);
            break;
        }
    }

    std::string result = detokenize_to_text(g_model, out);
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_resc_ai_skynetmonitor_service_NativeLlama_freeModel(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    free_all_locked();
}

}
