import {Router} from "express"

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
