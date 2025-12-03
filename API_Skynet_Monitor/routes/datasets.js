import { Router } from "express"

export default function datasetsRouter({ db }) {
    const router = Router()

    router.post("/", async (req, res) => {
        try {
            const { name, description, is_conversational } = req.body
            if (!name) return res.status(400).json({ error: "Missing dataset name" })

            const [result] = await db.execute(
                `INSERT INTO datasets (id_datatset, name, description, is_conversational)
                 VALUES (NULL, ?, ?, ?)`,
                [name, description || null, is_conversational ? 1 : 0]
            )

            res.json({
                message: "Dataset created successfully",
                id_datatset: result.insertId
            })
        } catch (err) {
            res.status(500).json({ error: err.message })
        }
    })

    router.get("/", async (req, res) => {
        try {
            const [datasets] = await db.execute("SELECT * FROM datasets")
            for (const dataset of datasets) {
                const [prompts] = await db.execute(
                    "SELECT * FROM prompts WHERE id_datatset = ?",
                    [dataset.id_datatset]
                )
                dataset.prompts = prompts
            }
            res.json(datasets)
        } catch (err) {
            res.status(500).json({ error: err.message })
        }
    })

    router.get("/:id", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT * FROM datasets WHERE id_datatset = ?",
                [req.params.id]
            )
            if (rows.length === 0) return res.status(404).json({ error: "Dataset not found" })

            const dataset = rows[0]
            const [prompts] = await db.execute(
                "SELECT * FROM prompts WHERE id_datatset = ?",
                [dataset.id_datatset]
            )
            dataset.prompts = prompts
            res.json(dataset)
        } catch (err) {
            res.status(500).json({ error: err.message })
        }
    })

    router.put("/:id", async (req, res) => {
        try {
            const { name, description, is_conversational } = req.body
            await db.execute(
                `UPDATE datasets
                 SET name = ?,
                     description = ?,
                     is_conversational = ?
                 WHERE id_datatset = ?`,
                [name, description, is_conversational ? 1 : 0, req.params.id]
            )
            res.json({ message: "Dataset updated successfully" })
        } catch (err) {
            res.status(500).json({ error: err.message })
        }
    })

    router.delete("/:id", async (req, res) => {
        try {
            await db.execute("DELETE FROM prompts WHERE id_datatset = ?", [req.params.id])
            await db.execute("DELETE FROM datasets WHERE id_datatset = ?", [req.params.id])
            res.json({ message: "Dataset and its prompts deleted successfully" })
        } catch (err) {
            res.status(500).json({ error: err.message })
        }
    })

    return router
}
