import {Router} from "express"

function parseBoolean(value) {
    if (typeof value === "boolean") return value
    if (typeof value === "number") return value === 1
    if (typeof value === "string") {
        const normalized = value.trim().toLowerCase()
        return normalized === "true" || normalized === "1"
    }
    return false
}

function readField(source, ...keys) {
    if (!source || typeof source !== "object" || Array.isArray(source)) return undefined
    for (const key of keys) {
        if (Object.prototype.hasOwnProperty.call(source, key)) return source[key]
    }
    return undefined
}

function parseNullableInteger(value) {
    if (value === undefined || value === null || value === "") return null
    const parsed = Number(value)
    if (!Number.isInteger(parsed) || parsed < 0) return null
    return parsed
}

function parseNullableFloat(value) {
    if (value === undefined || value === null || value === "") return null
    const parsed = Number(value)
    if (!Number.isFinite(parsed)) return null
    return parsed
}

function normalizePerformanceSample(sample) {
    return {
        sampleTimeMs: parseNullableInteger(readField(sample, "sampleTimeMs", "sample_time_ms", "timeMs", "time_ms")),
        batteryPercent: parseNullableFloat(readField(sample, "batteryPercent", "battery_percent")),
        ramCurrentMb: parseNullableFloat(readField(sample, "ramCurrentMb", "ram_current_mb", "ramUsedMb", "ram_used_mb")),
        ramMaxMb: parseNullableFloat(readField(sample, "ramMaxMb", "ram_max_mb", "ramTotalMb", "ram_total_mb")),
        batteryTemperatureC: parseNullableFloat(readField(
            sample,
            "batteryTemperatureC",
            "battery_temperature_c",
            "batteryTempC",
            "battery_temp_c"
        )),
    }
}

async function insertPerformanceSamples(db, idResult, rawSamples) {
    if (!Array.isArray(rawSamples) || rawSamples.length === 0) return 0

    const values = rawSamples
        .filter((sample) => sample && typeof sample === "object" && !Array.isArray(sample))
        .map((sample) => {
            const normalized = normalizePerformanceSample(sample)
            return [
                idResult,
                normalized.sampleTimeMs,
                normalized.batteryPercent,
                normalized.ramCurrentMb,
                normalized.ramMaxMb,
                normalized.batteryTemperatureC,
            ]
        })

    if (values.length === 0) return 0

    await db.query(
        `INSERT INTO prompt_result_device_performance
         (id_result, sample_time_ms, battery_percent, ram_current_mb, ram_max_mb, battery_temperature_c)
         VALUES ?`,
        [values]
    )

    return values.length
}

export default function promptsRouter({db}) {
    const router = Router()

    router.post("/", async (req, res) => {
        try {
            const {prompt, id_datatset} = req.body
            if (!prompt || !id_datatset)
                return res.status(400).json({error: "Missing prompt or dataset ID"})

            await db.execute(
                `INSERT INTO prompts (id_prompt, prompt, id_datatset)
                 VALUES (NULL, ?, ?)`,
                [prompt, id_datatset]
            )
            res.json({message: "Prompt created successfully"})
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/", async (req, res) => {
        try {
            const [rows] = await db.execute("SELECT * FROM prompts")
            res.json(rows)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.post("/results", async (req, res) => {
        try {
            const {
                response,
                id_prompt,
                id_model,
                id_devices,
                is_think,
                performance_samples,
                performanceSamples,
            } = req.body
            if (!response || !id_prompt || !id_model || !id_devices)
                return res.status(400).json({error: "Missing response, prompt ID, model ID, or device ID"})

            const responseTimeMs = parseNullableInteger(readField(
                req.body,
                "responseTimeMs",
                "response_time_ms",
                "durationMs",
                "duration_ms"
            ))
            const responseTokenCount = parseNullableInteger(readField(
                req.body,
                "responseTokenCount",
                "response_token_count",
                "tokenCount",
                "token_count",
                "tokens"
            ))
            const responseTokensPerS = parseNullableFloat(readField(
                req.body,
                "responseTokensPerS",
                "response_tokens_per_s",
                "tokensPerSecond",
                "tokens_per_second"
            ))

            const [result] = await db.execute(
                `INSERT INTO prompts_results
                 (id_result, response, is_think, response_time_ms, response_token_count, response_tokens_per_s,
                  id_prompt, id_model, id_devices)
                 VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)`,
                [
                    response,
                    parseBoolean(is_think) ? 1 : 0,
                    responseTimeMs,
                    responseTokenCount,
                    responseTokensPerS,
                    id_prompt,
                    id_model,
                    id_devices,
                ]
            )

            const insertedPerformanceSamples = await insertPerformanceSamples(
                db,
                result.insertId,
                performance_samples || performanceSamples
            )

            const [rows] = await db.execute(
                "SELECT * FROM prompts_results WHERE id_result = ?",
                [result.insertId]
            )
            res.status(201).json({
                ...rows[0],
                insertedPerformanceSamples,
            })
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/results", async (req, res) => {
        try {
            const [rows] = await db.execute("SELECT * FROM prompts_results ORDER BY id_result DESC")
            res.json(rows)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/results/:id", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT * FROM prompts_results WHERE id_result = ?",
                [req.params.id]
            )
            if (rows.length === 0) return res.status(404).json({error: "Prompt result not found"})
            res.json(rows[0])
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.post("/results/:id/performance", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT id_result FROM prompts_results WHERE id_result = ?",
                [req.params.id]
            )
            if (rows.length === 0) return res.status(404).json({error: "Prompt result not found"})

            const samples = Array.isArray(req.body) ? req.body : readField(req.body, "samples", "performance_samples")
            const insertedCount = await insertPerformanceSamples(db, req.params.id, samples)
            res.status(201).json({message: "Performance samples created", insertedCount})
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/results/:id/performance", async (req, res) => {
        try {
            const [rows] = await db.execute(
                `SELECT *
                 FROM prompt_result_device_performance
                 WHERE id_result = ?
                 ORDER BY sample_time_ms ASC, id_performance ASC`,
                [req.params.id]
            )
            res.json(rows)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/:id", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT * FROM prompts WHERE id_prompt = ?",
                [req.params.id]
            )
            if (rows.length === 0) return res.status(404).json({error: "Prompt not found"})
            res.json(rows[0])
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/:id/results", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT * FROM prompts_results WHERE id_prompt = ? ORDER BY id_result DESC",
                [req.params.id]
            )
            res.json(rows)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.put("/:id", async (req, res) => {
        try {
            const {prompt, id_datatset} = req.body
            await db.execute(
                `UPDATE prompts
                 SET prompt      = ?,
                     id_datatset = ?
                 WHERE id_prompt = ?`,
                [prompt, id_datatset, req.params.id]
            )
            res.json({message: "Prompt updated successfully"})
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.delete("/:id", async (req, res) => {
        try {
            await db.execute("DELETE FROM prompts WHERE id_prompt = ?", [req.params.id])
            res.json({message: "Prompt deleted successfully"})
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    return router
}
