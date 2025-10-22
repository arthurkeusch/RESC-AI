// cpp
#include <jni.h>
#include <string>
#include <vector>
#include <mutex>
#include <atomic>
#include <algorithm>
#include <cstdint>
#include <random>
#include <limits>
#include <android/log.h>
#include "llama.h"

#define LOG_TAG "SkynetNative"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR,  LOG_TAG, __VA_ARGS__)

// Configuration basique
static uint32_t CFG_N_CTX = 1024;
static uint32_t CFG_N_THREADS = 4;
static uint32_t CFG_N_THREADS_BATCH = 4;
static uint32_t CFG_N_BATCH = 512;
static uint32_t CFG_N_UBATCH = 32;
static uint32_t CFG_SEED = LLAMA_DEFAULT_SEED;
static bool     CFG_FORCE_ADD_BOS = true;
static int32_t  CFG_N_PREDICT = 128;
static float    CFG_TEMPERATURE = 0.20f;  // <= 0.0 => greedy
static int32_t  CFG_TOP_K = 30;           // <= 0 => désactivé
static float    CFG_TOP_P = 0.90f;        // 0..1
static float    CFG_REPEAT_PENALTY = 1.10f;
static size_t   CFG_REPEAT_LAST_N = 64;

// État global
static std::mutex g_mutex;
static std::atomic<bool> g_backend_init{false};
static llama_model *g_model = nullptr;
static const llama_vocab *g_vocab = nullptr;
static llama_context *g_ctx = nullptr;
static std::vector<llama_token> g_last_tokens;
static uint32_t g_n_vocab = 0;
static std::mt19937 g_rng;

static void free_all_locked() {
    if (g_ctx) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_vocab = nullptr;
    g_last_tokens.clear();
    g_n_vocab = 0;
}

static void ensure_backend() {
    bool expected = false;
    if (g_backend_init.compare_exchange_strong(expected, true)) {
        llama_backend_init();
    }
}

static std::vector<llama_token> tokenize_full(const llama_vocab * vocab,
                                              const std::string &text,
                                              bool add_bos,
                                              bool parse_special) {
    const int32_t need = llama_tokenize(vocab,
                                        text.c_str(),
                                        (int32_t) text.size(),
                                        nullptr,
                                        0,
            /*add_special*/ add_bos,
            /*parse_special*/ parse_special);
    if (need < 0) {
        LOGE("tokenize failed (pre-size): %d", need);
        return {};
    }

    std::vector<llama_token> out((size_t) need);
    const int32_t n = llama_tokenize(vocab,
                                     text.c_str(),
                                     (int32_t) text.size(),
                                     out.data(),
                                     (int32_t) out.size(),
            /*add_special*/ add_bos,
            /*parse_special*/ parse_special);
    if (n < 0) {
        LOGE("tokenize failed: %d", n);
        return {};
    }
    return out;
}

static bool decode_prompt(llama_context *ctx,
                          const std::vector<llama_token> &tokens,
                          uint32_t &n_past) {
    for (int32_t i = 0; i < (int32_t) tokens.size();) {
        const int32_t n = std::min<int32_t>((int32_t) CFG_N_UBATCH,
                                            (int32_t) tokens.size() - i);

        llama_batch batch = llama_batch_init(n, 0, 1);

        for (int32_t j = 0; j < n; ++j) {
            batch.token[j]  = tokens[i + j];
            batch.pos[j]    = (llama_pos) (n_past + j);
            batch.seq_id[j] = 0;
            batch.logits[j] = (j == n - 1) ? 1 : 0;
        }

        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);

        if (rc != 0) {
            LOGE("llama_decode(prompt) failed: %d", rc);
            return false;
        }

        n_past += (uint32_t) n;
        i += n;
    }
    return true;
}

static inline void apply_repeat_penalty(std::vector<float> &logits,
                                        const std::vector<llama_token> &last_tokens,
                                        size_t last_n,
                                        float penalty) {
    if (penalty <= 1.0f || last_tokens.empty()) return;
    const size_t n = std::min(last_tokens.size(), last_n);
    for (size_t i = last_tokens.size() - n; i < last_tokens.size(); ++i) {
        const llama_token t = last_tokens[i];
        if ((size_t) t >= logits.size()) continue;
        float &logit = logits[(size_t) t];
        if (logit > 0.0f) logit /= penalty;
        else              logit *= penalty;
    }
}

static llama_token sample_next(llama_context *ctx, const llama_vocab * vocab,
                               const std::vector<llama_token> &last_tokens,
                               bool ban_eos) {
    const float *logits_ptr = llama_get_logits(ctx);
    if (!logits_ptr) {
        LOGE("logits null");
        return 0;
    }

    // Copie des logits
    std::vector<float> logits(g_n_vocab);
    std::copy(logits_ptr, logits_ptr + g_n_vocab, logits.begin());

    // Ban EOS si demandé
    if (ban_eos) {
        const llama_token eos = llama_vocab_eos(vocab);
        if (eos >= 0 && (uint32_t) eos < g_n_vocab) {
            logits[(size_t) eos] = -std::numeric_limits<float>::infinity();
        }
    }

    // Pénalité de répétition
    apply_repeat_penalty(logits, last_tokens, CFG_REPEAT_LAST_N, CFG_REPEAT_PENALTY);

    // Greedy si temp <= 0
    if (CFG_TEMPERATURE <= 0.0f) {
        auto it = std::max_element(logits.begin(), logits.end());
        return (llama_token) std::distance(logits.begin(), it);
    }

    // Température
    const float inv_temp = 1.0f / std::max(1e-8f, CFG_TEMPERATURE);
    for (float &x : logits) x *= inv_temp;

    // Top-k
    if (CFG_TOP_K > 0 && (uint32_t) CFG_TOP_K < g_n_vocab) {
        // Trouver le seuil top-k
        std::vector<float> tmp = logits;
        std::nth_element(tmp.begin(), tmp.begin() + CFG_TOP_K, tmp.end(), std::greater<float>());
        const float kth = tmp[CFG_TOP_K];
        for (float &x : logits) if (x < kth) x = -std::numeric_limits<float>::infinity();
    }

    // Softmax partielle pour top-p
    // 1) trier par logit desc
    std::vector<int> idx(g_n_vocab);
    for (uint32_t i = 0; i < g_n_vocab; ++i) idx[i] = (int) i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return logits[(size_t) a] > logits[(size_t) b];
    });

    // 2) convertir en proba pour accumuler
    const float max_logit = logits[(size_t) idx[0]];
    std::vector<float> probs(g_n_vocab, 0.0f);
    float sum = 0.0f;
    for (uint32_t i = 0; i < g_n_vocab; ++i) {
        const float v = logits[(size_t) idx[i]];
        if (v == -std::numeric_limits<float>::infinity()) break;
        const float p = std::exp(v - max_logit);
        probs[(size_t) idx[i]] = p;
        sum += p;
    }
    if (sum <= 0.0f) {
        // fallback greedy
        auto it = std::max_element(logits.begin(), logits.end());
        return (llama_token) std::distance(logits.begin(), it);
    }
    for (float &p : probs) p /= sum;

    // 3) top-p: cumuler selon l'ordre trié et couper
    if (CFG_TOP_P > 0.0f && CFG_TOP_P < 1.0f) {
        float cum = 0.0f;
        for (uint32_t i = 0; i < g_n_vocab; ++i) {
            const int id = idx[i];
            const float p = probs[(size_t) id];
            if (p == 0.0f) break;
            cum += p;
            if (cum > CFG_TOP_P) {
                for (uint32_t j = i + 1; j < g_n_vocab; ++j) {
                    probs[(size_t) idx[j]] = 0.0f;
                }
                break;
            }
        }
        // renormaliser
        float s = 0.0f;
        for (float p : probs) s += p;
        if (s > 0.0f) for (float &p : probs) p /= s;
    }

    // Échantillonnage multinomial
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(g_rng);
    float acc = 0.0f;
    for (uint32_t i = 0; i < g_n_vocab; ++i) {
        acc += probs[i];
        if (r <= acc) return (llama_token) i;
    }
    // fallback: dernier
    return (llama_token) (g_n_vocab - 1);
}

static std::string detokenize_to_text(const llama_vocab * vocab,
                                      const std::vector<llama_token> &toks) {
    std::string out;
    out.reserve(toks.size() * 4);
    char buf[8192];

    for (llama_token t: toks) {
        int n = llama_token_to_piece(vocab, t, buf, (int32_t) sizeof(buf), /*lstrip*/ 0, /*special*/ false);
        if (n <= 0) continue;
        out.append(buf, (size_t) n);
    }

    // Nettoyage simple des contrôles non imprimables (hors CR/LF/TAB)
    std::string safe;
    safe.reserve(out.size());
    for (char c : out) {
        if ((c >= 32 && c != 127) || c == '\n' || c == '\r' || c == '\t') safe.push_back(c);
    }
    return safe;
}

extern "C" {

JNIEXPORT jboolean JNICALL
Java_resc_ai_skynetmonitor_service_NativeLlama_initModel(JNIEnv *env, jobject /*thiz*/,
                                                         jstring jModelPath) {
    const char *cpath = env->GetStringUTFChars(jModelPath, nullptr);
    LOGD("initModel called with %s", cpath ? cpath : "(null)");

    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_backend();
    free_all_locked();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(cpath, mparams);
    env->ReleaseStringUTFChars(jModelPath, cpath);

    if (!g_model) {
        LOGE("failed to load model");
        return JNI_FALSE;
    }

    g_vocab = llama_model_get_vocab(g_model);
    if (!g_vocab) {
        LOGE("failed to get vocab");
        free_all_locked();
        return JNI_FALSE;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = CFG_N_CTX;
    cparams.n_batch        = CFG_N_BATCH;
    cparams.n_ubatch       = CFG_N_UBATCH;
    cparams.n_threads      = CFG_N_THREADS;
    cparams.n_threads_batch= CFG_N_THREADS_BATCH;
    cparams.type_k         = GGML_TYPE_F16; // laisser par défaut si souhaité
    cparams.type_v         = GGML_TYPE_F16;

    g_ctx = llama_init_from_model(g_model, cparams);
    if (!g_ctx) {
        LOGE("failed to create context");
        free_all_locked();
        return JNI_FALSE;
    }

    g_n_vocab = (uint32_t) llama_vocab_n_tokens(g_vocab);
    g_rng.seed(CFG_SEED);

    LOGD("Model READY. vocab=%u ctx=%d", g_n_vocab, llama_n_ctx(g_ctx));
    return JNI_TRUE;
}

JNIEXPORT jstring JNICALL
Java_resc_ai_skynetmonitor_service_NativeLlama_generate(JNIEnv *env, jobject /*thiz*/,
                                                        jstring jPrompt) {
    const char *cprompt = env->GetStringUTFChars(jPrompt, nullptr);
    std::string user_prompt(cprompt ? cprompt : "");
    env->ReleaseStringUTFChars(jPrompt, cprompt);

    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_model || !g_ctx || !g_vocab) {
        const char *msg = "Model not initialized";
        return env->NewStringUTF(msg);
    }

    // Clear KV cache
    // TODO:  llama_kv_cache_seq_rm(g_ctx, /*seq_id*/ -1, /*p0*/ 0, /*p1*/ -1);
    g_last_tokens.clear();

    const bool add_bos = CFG_FORCE_ADD_BOS || llama_vocab_get_add_bos(g_vocab);
    std::vector<llama_token> tokens = tokenize_full(g_vocab, user_prompt, add_bos, /*parse_special*/ false);
    if (tokens.empty()) {
        const char *msg = "Tokenization failed or empty prompt";
        return env->NewStringUTF(msg);
    }

    uint32_t n_past = 0;
    if (!decode_prompt(g_ctx, tokens, n_past)) {
        const char *msg = "Prompt decode failed";
        return env->NewStringUTF(msg);
    }

    g_last_tokens = tokens;

    std::vector<llama_token> out;
    out.reserve((size_t) CFG_N_PREDICT);

    for (int i = 0; i < CFG_N_PREDICT; ++i) {
        const llama_token next = sample_next(g_ctx, g_vocab, g_last_tokens, /*ban_eos*/ false);
        const llama_token eos  = llama_vocab_eos(g_vocab);
        if (next == eos) break;

        out.push_back(next);
        g_last_tokens.push_back(next);

        // Feed the sampled token
        llama_batch batch = llama_batch_init(1, 0, 1);
        batch.token[0]  = next;
        batch.pos[0]    = (llama_pos) n_past;
        batch.seq_id[0] = 0;
        batch.logits[0] = 1;

        const int rc = llama_decode(g_ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            LOGE("llama_decode(gen) failed: %d", rc);
            break;
        }

        n_past += 1;
    }

    std::string result = detokenize_to_text(g_vocab, out);
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT void JNICALL
Java_resc_ai_skynetmonitor_service_NativeLlama_freeModel(JNIEnv *, jobject) {
    std::lock_guard<std::mutex> lock(g_mutex);
    free_all_locked();
}

} // extern "C"
