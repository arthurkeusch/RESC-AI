import {Router} from "express"

function normalizeName(value) {
    if (typeof value !== "string") return ""
    return value.trim()
}

export default function modelsRouter({db}) {
    const router = Router()

    async function findModelByName(name) {
        const [rows] = await db.execute(
            "SELECT id_model, name FROM models WHERE name = ? LIMIT 1",
            [name]
        )
        return rows[0] || null
    }

    async function findModelById(id) {
        const [rows] = await db.execute(
            "SELECT id_model, name FROM models WHERE id_model = ?",
            [id]
        )
        return rows[0] || null
    }

    router.get("/", async (req, res) => {
        try {
            const name = normalizeName(req.query.name)
            if (name) {
                const model = await findModelByName(name)
                if (!model) return res.status(404).json({error: "Model not found"})
                return res.json(model)
            }

            const [rows] = await db.execute("SELECT id_model, name FROM models ORDER BY name ASC")
            res.json(rows)
        } catch (err) {
            res.status(500).json({error: "Failed to fetch models: " + err.message})
        }
    })

    router.post("/", async (req, res) => {
        try {
            const name = normalizeName(req.body?.name)
            if (!name) return res.status(400).json({error: "Missing model name"})

            const existing = await findModelByName(name)
            if (existing) return res.json(existing)

            const [result] = await db.execute(
                "INSERT INTO models (id_model, name) VALUES (NULL, ?)",
                [name]
            )

            const created = await findModelById(result.insertId)
            res.status(201).json(created)
        } catch (err) {
            res.status(500).json({error: "Failed to create model: " + err.message})
        }
    })

    router.get("/:id", async (req, res) => {
        try {
            const model = await findModelById(req.params.id)
            if (!model) return res.status(404).json({error: "Model not found"})
            res.json(model)
        } catch (err) {
            res.status(500).json({error: "Failed to fetch model: " + err.message})
        }
    })

    router.put("/:id", async (req, res) => {
        try {
            const name = normalizeName(req.body?.name)
            if (!name) return res.status(400).json({error: "Missing model name"})

            const model = await findModelById(req.params.id)
            if (!model) return res.status(404).json({error: "Model not found"})

            await db.execute(
                "UPDATE models SET name = ? WHERE id_model = ?",
                [name, req.params.id]
            )

            const updated = await findModelById(req.params.id)
            res.json(updated)
        } catch (err) {
            res.status(500).json({error: "Failed to update model: " + err.message})
        }
    })

    router.delete("/:id", async (req, res) => {
        try {
            const model = await findModelById(req.params.id)
            if (!model) return res.status(404).json({error: "Model not found"})

            await db.execute("DELETE FROM models WHERE id_model = ?", [req.params.id])
            res.json({message: "Model deleted", id_model: Number(req.params.id)})
        } catch (err) {
            res.status(500).json({error: "Failed to delete model: " + err.message})
        }
    })

    return router
}
