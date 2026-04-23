import {Router} from "express"

function parseOptionalPositiveInt(value) {
    if (value === undefined || value === null || value === "") return null
    const parsed = Number(value)
    if (!Number.isInteger(parsed) || parsed <= 0) return null
    return parsed
}

function deviceFilterClause(deviceId, alias = "r") {
    return deviceId ? ` AND ${alias}.id_devices = ?` : ""
}

export default function statsRouter({db}) {
    const router = Router()

    router.get("/models", async (req, res) => {
        try {
            const deviceId = parseOptionalPositiveInt(req.query.deviceId || req.query.id_devices)
            const params = deviceId ? [deviceId] : []

            const [rows] = await db.execute(
                `SELECT m.id_model,
                        m.name,
                        COUNT(r.id_result)                       AS result_count,
                        AVG(r.response_tokens_per_s)             AS avg_tokens_per_s,
                        AVG(r.response_time_ms)                  AS avg_response_time_ms,
                        AVG(r.response_token_count)              AS avg_response_token_count
                 FROM models m
                          LEFT JOIN prompts_results r ON r.id_model = m.id_model${deviceFilterClause(deviceId)}
                 GROUP BY m.id_model, m.name
                 ORDER BY avg_tokens_per_s DESC, m.name ASC`,
                params
            )

            res.json(rows)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/datasets/:id/summary", async (req, res) => {
        try {
            const deviceId = parseOptionalPositiveInt(req.query.deviceId || req.query.id_devices)
            const params = deviceId ? [deviceId, req.params.id] : [req.params.id]

            const [rows] = await db.execute(
                `SELECT d.id_datatset,
                        d.name,
                        COUNT(r.id_result)                       AS result_count,
                        AVG(r.response_tokens_per_s)             AS avg_tokens_per_s,
                        AVG(r.response_time_ms)                  AS avg_response_time_ms,
                        AVG(r.response_token_count)              AS avg_response_token_count
                 FROM datasets d
                          LEFT JOIN prompts p ON p.id_datatset = d.id_datatset
                          LEFT JOIN prompts_results r ON r.id_prompt = p.id_prompt${deviceFilterClause(deviceId)}
                 WHERE d.id_datatset = ?
                 GROUP BY d.id_datatset, d.name`,
                params
            )

            if (rows.length === 0) return res.status(404).json({error: "Dataset not found"})
            res.json(rows[0])
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/datasets/:id/models", async (req, res) => {
        try {
            const deviceId = parseOptionalPositiveInt(req.query.deviceId || req.query.id_devices)
            const params = [req.params.id]
            if (deviceId) params.push(deviceId)

            const [rows] = await db.execute(
                `SELECT m.id_model,
                        m.name,
                        COUNT(r.id_result)                       AS result_count,
                        AVG(r.response_tokens_per_s)             AS avg_tokens_per_s,
                        AVG(r.response_time_ms)                  AS avg_response_time_ms,
                        AVG(r.response_token_count)              AS avg_response_token_count
                 FROM prompts_results r
                          INNER JOIN prompts p ON p.id_prompt = r.id_prompt
                          INNER JOIN models m ON m.id_model = r.id_model
                 WHERE p.id_datatset = ?${deviceFilterClause(deviceId)}
                 GROUP BY m.id_model, m.name
                 ORDER BY avg_tokens_per_s DESC, m.name ASC`,
                params
            )

            res.json(rows)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/datasets/:id/results", async (req, res) => {
        try {
            const deviceId = parseOptionalPositiveInt(req.query.deviceId || req.query.id_devices)
            const params = [req.params.id]
            if (deviceId) params.push(deviceId)

            const [rows] = await db.execute(
                `SELECT r.*
                 FROM prompts_results r
                          INNER JOIN prompts p ON p.id_prompt = r.id_prompt
                 WHERE p.id_datatset = ?${deviceFilterClause(deviceId)}
                 ORDER BY r.id_result DESC`,
                params
            )

            res.json(rows)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/datasets/:id/latest-performance", async (req, res) => {
        try {
            const deviceId = parseOptionalPositiveInt(req.query.deviceId || req.query.id_devices)
            const params = [req.params.id]
            if (deviceId) params.push(deviceId)

            const [latestRows] = await db.execute(
                `SELECT r.id_result
                 FROM prompts_results r
                          INNER JOIN prompts p ON p.id_prompt = r.id_prompt
                 WHERE p.id_datatset = ?${deviceFilterClause(deviceId)}
                 ORDER BY r.id_result DESC
                 LIMIT 1`,
                params
            )

            if (latestRows.length === 0) return res.json([])

            const [samples] = await db.execute(
                `SELECT *
                 FROM prompt_result_device_performance
                 WHERE id_result = ?
                 ORDER BY sample_time_ms ASC, id_performance ASC`,
                [latestRows[0].id_result]
            )

            res.json(samples)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    return router
}
