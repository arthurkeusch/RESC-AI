import {Router} from "express"

function buildDeviceName(body) {
    if (body.name && typeof body.name === "string") return body.name.trim()

    const parts = [
        body.brand,
        body.model,
        body.android_version ? `Android ${body.android_version}` : null,
        body.soc,
    ].filter(Boolean)

    return parts.join(" - ").trim()
}

export default function devicesRouter({db}) {
    const router = Router()

    router.post("/", async (req, res) => {
        try {
            const name = buildDeviceName(req.body || {})
            if (!name) return res.status(400).json({error: "Missing device name or metadata"})

            const [existing] = await db.execute(
                "SELECT * FROM devices WHERE name = ? LIMIT 1",
                [name]
            )

            if (existing.length > 0) {
                return res.json(existing[0])
            }

            const [result] = await db.execute(
                "INSERT INTO devices (id_devices, name) VALUES (NULL, ?)",
                [name]
            )

            const [created] = await db.execute(
                "SELECT * FROM devices WHERE id_devices = ?",
                [result.insertId]
            )

            res.status(201).json(created[0])
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/", async (req, res) => {
        try {
            const [rows] = await db.execute("SELECT * FROM devices")
            res.json(rows)
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    router.get("/:id", async (req, res) => {
        try {
            const [rows] = await db.execute(
                "SELECT * FROM devices WHERE id_devices = ?",
                [req.params.id]
            )
            if (rows.length === 0) return res.status(404).json({error: "Device not found"})
            res.json(rows[0])
        } catch (err) {
            res.status(500).json({error: err.message})
        }
    })

    return router
}
