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
            const {response, id_prompt, id_model, id_devices, is_think} = req.body
            if (!response || !id_prompt || !id_model || !id_devices)
                return res.status(400).json({error: "Missing response, prompt ID, model ID, or device ID"})

            const [result] = await db.execute(
                `INSERT INTO prompts_results (id_result, response, is_think, id_prompt, id_model, id_devices)
                 VALUES (NULL, ?, ?, ?, ?, ?)`,
                [response, parseBoolean(is_think) ? 1 : 0, id_prompt, id_model, id_devices]
            )

            const [rows] = await db.execute(
                "SELECT * FROM prompts_results WHERE id_result = ?",
                [result.insertId]
            )
            res.status(201).json(rows[0])
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/results", async (req, res) => {
        try {
            const [rows] = await db.execute("SELECT * FROM prompts_results")
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
                "SELECT * FROM prompts_results WHERE id_prompt = ?",
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
