import {Router} from "express"

const ALLOWED_STATUSES = new Set(["recording", "completed"])

function httpError(statusCode, message) {
    const error = new Error(message)
    error.statusCode = statusCode
    return error
}

function hasOwn(source, key) {
    return Object.prototype.hasOwnProperty.call(source, key)
}

function hasField(source, ...keys) {
    if (!source || typeof source !== "object" || Array.isArray(source)) return false
    return keys.some((key) => hasOwn(source, key))
}

function readField(source, ...keys) {
    if (!source || typeof source !== "object" || Array.isArray(source)) return undefined
    for (const key of keys) {
        if (hasOwn(source, key)) return source[key]
    }
    return undefined
}

function parseId(value, fieldName = "id") {
    const parsed = Number(value)
    if (!Number.isInteger(parsed) || parsed <= 0) {
        throw httpError(400, `${fieldName} must be a positive integer`)
    }
    return parsed
}

function parseBoolean(value, fieldName) {
    if (typeof value === "boolean") return value
    if (typeof value === "number") {
        if (value === 1) return true
        if (value === 0) return false
    }
    if (typeof value === "string") {
        const normalized = value.trim().toLowerCase()
        if (normalized === "true" || normalized === "1") return true
        if (normalized === "false" || normalized === "0") return false
    }
    throw httpError(400, `${fieldName} must be a boolean`)
}

function parseNullableInteger(value, fieldName) {
    if (value === undefined) return undefined
    if (value === null || value === "") return null

    const parsed = Number(value)
    if (!Number.isInteger(parsed) || parsed < 0) {
        throw httpError(400, `${fieldName} must be a non-negative integer`)
    }
    return parsed
}

function parseNullableFloat(value, fieldName) {
    if (value === undefined) return undefined
    if (value === null || value === "") return null

    const parsed = Number(value)
    if (!Number.isFinite(parsed)) {
        throw httpError(400, `${fieldName} must be a valid number`)
    }
    return parsed
}

function parseNullableString(value, fieldName, maxLength = 255) {
    if (value === undefined) return undefined
    if (value === null) return null
    if (typeof value !== "string") {
        throw httpError(400, `${fieldName} must be a string`)
    }

    const trimmed = value.trim()
    if (!trimmed) return null
    if (trimmed.length > maxLength) {
        throw httpError(400, `${fieldName} must be at most ${maxLength} characters`)
    }
    return trimmed
}

function parseStatus(value) {
    if (value === undefined) return undefined
    if (typeof value !== "string") {
        throw httpError(400, "status must be a string")
    }

    const normalized = value.trim().toLowerCase()
    if (!ALLOWED_STATUSES.has(normalized)) {
        throw httpError(400, "status must be one of: recording, completed")
    }
    return normalized
}

function isTruthyFlag(value) {
    if (typeof value === "boolean") return value
    if (typeof value === "number") return value === 1
    if (typeof value === "string") {
        const normalized = value.trim().toLowerCase()
        return normalized === "true" || normalized === "1"
    }
    return false
}

function normalizeRecordingInput(body = {}) {
    return {
        status: parseStatus(readField(body, "status")),
        startedAtMillis: hasField(body, "startedAtMillis", "started_at_millis")
            ? parseNullableInteger(readField(body, "startedAtMillis", "started_at_millis"), "startedAtMillis")
            : undefined,
        endedAtMillis: hasField(body, "endedAtMillis", "ended_at_millis")
            ? parseNullableInteger(readField(body, "endedAtMillis", "ended_at_millis"), "endedAtMillis")
            : undefined,
        isFall: hasField(body, "isFall", "is_fall")
            ? parseBoolean(readField(body, "isFall", "is_fall"), "isFall")
            : undefined,
        fallStartMillis: hasField(body, "fallStartMillis", "fall_start_millis")
            ? parseNullableInteger(readField(body, "fallStartMillis", "fall_start_millis"), "fallStartMillis")
            : undefined,
        fallEndMillis: hasField(body, "fallEndMillis", "fall_end_millis")
            ? parseNullableInteger(readField(body, "fallEndMillis", "fall_end_millis"), "fallEndMillis")
            : undefined,
    }
}

function computeAccelerationNorm(ax, ay, az) {
    if (ax == null || ay == null || az == null) return null
    return Math.sqrt((ax * ax) + (ay * ay) + (az * az))
}

function normalizePoint(point, index) {
    if (!point || typeof point !== "object" || Array.isArray(point)) {
        throw httpError(400, `points[${index}] must be an object`)
    }

    const wallTimeMillis = parseNullableInteger(
        readField(point, "wallTimeMillis", "wall_time_millis"),
        `points[${index}].wallTimeMillis`
    )

    if (wallTimeMillis == null) {
        throw httpError(400, `points[${index}].wallTimeMillis is required`)
    }

    const ax = hasField(point, "ax")
        ? parseNullableFloat(readField(point, "ax"), `points[${index}].ax`)
        : null
    const ay = hasField(point, "ay")
        ? parseNullableFloat(readField(point, "ay"), `points[${index}].ay`)
        : null
    const az = hasField(point, "az")
        ? parseNullableFloat(readField(point, "az"), `points[${index}].az`)
        : null

    const aNorm = hasField(point, "aNorm", "a_norm")
        ? parseNullableFloat(readField(point, "aNorm", "a_norm"), `points[${index}].aNorm`)
        : computeAccelerationNorm(ax, ay, az)

    return {
        wallTimeMillis,
        timeText: hasField(point, "timeText", "time_text")
            ? parseNullableString(readField(point, "timeText", "time_text"), `points[${index}].timeText`, 32)
            : null,
        ax,
        ay,
        az,
        aNorm,
        gx: hasField(point, "gx")
            ? parseNullableFloat(readField(point, "gx"), `points[${index}].gx`)
            : null,
        gy: hasField(point, "gy")
            ? parseNullableFloat(readField(point, "gy"), `points[${index}].gy`)
            : null,
        gz: hasField(point, "gz")
            ? parseNullableFloat(readField(point, "gz"), `points[${index}].gz`)
            : null,
        qw: hasField(point, "qw")
            ? parseNullableFloat(readField(point, "qw"), `points[${index}].qw`)
            : null,
        qx: hasField(point, "qx")
            ? parseNullableFloat(readField(point, "qx"), `points[${index}].qx`)
            : null,
        qy: hasField(point, "qy")
            ? parseNullableFloat(readField(point, "qy"), `points[${index}].qy`)
            : null,
        qz: hasField(point, "qz")
            ? parseNullableFloat(readField(point, "qz"), `points[${index}].qz`)
            : null,
        vax: hasField(point, "vax")
            ? parseNullableFloat(readField(point, "vax"), `points[${index}].vax`)
            : null,
        vay: hasField(point, "vay")
            ? parseNullableFloat(readField(point, "vay"), `points[${index}].vay`)
            : null,
        vaz: hasField(point, "vaz")
            ? parseNullableFloat(readField(point, "vaz"), `points[${index}].vaz`)
            : null,
    }
}

function normalizePointsPayload(body) {
    let rawPoints

    if (Array.isArray(body)) {
        rawPoints = body
    } else if (body && typeof body === "object") {
        const nestedPoints = readField(body, "points", "sensorPoints", "samples")
        rawPoints = Array.isArray(nestedPoints) ? nestedPoints : [body]
    } else {
        throw httpError(400, "Request body must contain a point object or an array of points")
    }

    if (rawPoints.length === 0) {
        throw httpError(400, "At least one sensor point is required")
    }

    return rawPoints.map((point, index) => normalizePoint(point, index))
}

function validateRecordingState(recording) {
    if (!ALLOWED_STATUSES.has(recording.status)) {
        throw httpError(400, "status must be one of: recording, completed")
    }

    if (recording.status === "completed") {
        if (recording.startedAtMillis == null || recording.endedAtMillis == null) {
            throw httpError(400, "A completed recording requires both startedAtMillis and endedAtMillis")
        }
    }

    if (
        recording.startedAtMillis != null &&
        recording.endedAtMillis != null &&
        recording.endedAtMillis < recording.startedAtMillis
    ) {
        throw httpError(400, "endedAtMillis must be greater than or equal to startedAtMillis")
    }

    if (!recording.isFall) {
        recording.fallStartMillis = null
        recording.fallEndMillis = null
        return recording
    }

    if (recording.fallEndMillis != null && recording.fallStartMillis == null) {
        throw httpError(400, "fallEndMillis requires fallStartMillis")
    }

    if (
        recording.fallStartMillis != null &&
        recording.fallEndMillis != null &&
        recording.fallEndMillis < recording.fallStartMillis
    ) {
        throw httpError(400, "fallEndMillis must be greater than or equal to fallStartMillis")
    }

    if (
        recording.startedAtMillis != null &&
        recording.fallStartMillis != null &&
        recording.fallStartMillis < recording.startedAtMillis
    ) {
        throw httpError(400, "fallStartMillis must be inside the recording range")
    }

    if (
        recording.endedAtMillis != null &&
        recording.fallEndMillis != null &&
        recording.fallEndMillis > recording.endedAtMillis
    ) {
        throw httpError(400, "fallEndMillis must be inside the recording range")
    }

    return recording
}

function formatRecording(row) {
    if (!row) return null

    const formatted = {
        idRecording: row.id_recording,
        startedAtMillis: row.started_at_millis,
        endedAtMillis: row.ended_at_millis,
        status: row.status,
        isFall: Boolean(row.is_fall),
        fallStartMillis: row.fall_start_millis,
        fallEndMillis: row.fall_end_millis,
        createdAt: row.created_at,
        updatedAt: row.updated_at,
    }

    if (row.points_count !== undefined) {
        formatted.pointsCount = Number(row.points_count)
    }

    return formatted
}

function formatPoint(row) {
    return {
        idSensorPoint: row.id_sensor_point,
        idRecording: row.id_recording,
        wallTimeMillis: row.wall_time_millis,
        timeText: row.time_text,
        ax: row.ax,
        ay: row.ay,
        az: row.az,
        aNorm: row.a_norm,
        gx: row.gx,
        gy: row.gy,
        gz: row.gz,
        qw: row.qw,
        qx: row.qx,
        qy: row.qy,
        qz: row.qz,
        vax: row.vax,
        vay: row.vay,
        vaz: row.vaz,
        createdAt: row.created_at,
    }
}

export default function heimdallRouter({db}) {
    const router = Router()

    async function getRecordingRow(idRecording) {
        const [rows] = await db.execute(
            "SELECT * FROM heimdall_recordings WHERE id_recording = ?",
            [idRecording]
        )
        return rows[0] || null
    }

    async function getRecordingBounds(idRecording) {
        const [rows] = await db.execute(
            `SELECT MIN(wall_time_millis) AS started_at_millis,
                    MAX(wall_time_millis) AS ended_at_millis,
                    COUNT(*)              AS points_count
             FROM heimdall_sensor_points
             WHERE id_recording = ?`,
            [idRecording]
        )
        return rows[0]
    }

    async function getRecordingById(idRecording, includePoints = false) {
        const [rows] = await db.execute(
            `SELECT r.*,
                    COALESCE(points.points_count, 0) AS points_count
             FROM heimdall_recordings r
                      LEFT JOIN (SELECT id_recording, COUNT(*) AS points_count
                                 FROM heimdall_sensor_points
                                 GROUP BY id_recording) points ON points.id_recording = r.id_recording
             WHERE r.id_recording = ?`,
            [idRecording]
        )

        if (rows.length === 0) return null

        const recording = formatRecording(rows[0])

        if (includePoints) {
            const [points] = await db.execute(
                `SELECT *
                 FROM heimdall_sensor_points
                 WHERE id_recording = ?
                 ORDER BY wall_time_millis ASC, id_sensor_point ASC`,
                [idRecording]
            )
            recording.points = points.map(formatPoint)
        }

        return recording
    }

    async function saveRecording(idRecording, recording) {
        await db.execute(
            `UPDATE heimdall_recordings
             SET started_at_millis = ?,
                 ended_at_millis   = ?,
                 status            = ?,
                 is_fall           = ?,
                 fall_start_millis = ?,
                 fall_end_millis   = ?
             WHERE id_recording = ?`,
            [
                recording.startedAtMillis,
                recording.endedAtMillis,
                recording.status,
                recording.isFall ? 1 : 0,
                recording.fallStartMillis,
                recording.fallEndMillis,
                idRecording,
            ]
        )
    }

    function handleError(res, err, fallbackMessage) {
        res.status(err.statusCode || 500).json({error: err.message || fallbackMessage})
    }

    router.post("/", async (req, res) => {
        try {
            const input = normalizeRecordingInput(req.body || {})
            const recording = validateRecordingState({
                status: input.status || "recording",
                startedAtMillis: input.startedAtMillis ?? null,
                endedAtMillis: input.endedAtMillis ?? null,
                isFall: input.isFall ?? false,
                fallStartMillis: input.fallStartMillis ?? null,
                fallEndMillis: input.fallEndMillis ?? null,
            })

            const [result] = await db.execute(
                `INSERT INTO heimdall_recordings
                 (id_recording, started_at_millis, ended_at_millis, status, is_fall, fall_start_millis, fall_end_millis)
                 VALUES (NULL, ?, ?, ?, ?, ?, ?)`,
                [
                    recording.startedAtMillis,
                    recording.endedAtMillis,
                    recording.status,
                    recording.isFall ? 1 : 0,
                    recording.fallStartMillis,
                    recording.fallEndMillis,
                ]
            )

            const created = await getRecordingById(result.insertId)
            res.status(201).json(created)
        } catch (err) {
            handleError(res, err, "Failed to create recording")
        }
    })

    router.get("/", async (req, res) => {
        try {
            const [rows] = await db.execute(
                `SELECT r.*,
                        COALESCE(points.points_count, 0) AS points_count
                 FROM heimdall_recordings r
                          LEFT JOIN (SELECT id_recording, COUNT(*) AS points_count
                                     FROM heimdall_sensor_points
                                     GROUP BY id_recording) points ON points.id_recording = r.id_recording
                 ORDER BY r.created_at DESC, r.id_recording DESC`
            )

            res.json(rows.map(formatRecording))
        } catch (err) {
            handleError(res, err, "Failed to fetch recordings")
        }
    })

    router.get("/:id", async (req, res) => {
        try {
            const idRecording = parseId(req.params.id, "id")
            const includePoints = isTruthyFlag(req.query.includePoints)

            const recording = await getRecordingById(idRecording, includePoints)
            if (!recording) {
                return res.status(404).json({error: "Recording not found"})
            }

            res.json(recording)
        } catch (err) {
            handleError(res, err, "Failed to fetch recording")
        }
    })

    const updateRecordingHandler = async (req, res) => {
        try {
            const idRecording = parseId(req.params.id, "id")
            const current = await getRecordingRow(idRecording)

            if (!current) {
                return res.status(404).json({error: "Recording not found"})
            }

            const input = normalizeRecordingInput(req.body || {})
            const nextRecording = validateRecordingState({
                status: input.status ?? current.status,
                startedAtMillis: input.startedAtMillis !== undefined ? input.startedAtMillis : current.started_at_millis,
                endedAtMillis: input.endedAtMillis !== undefined ? input.endedAtMillis : current.ended_at_millis,
                isFall: input.isFall !== undefined ? input.isFall : Boolean(current.is_fall),
                fallStartMillis: input.fallStartMillis !== undefined ? input.fallStartMillis : current.fall_start_millis,
                fallEndMillis: input.fallEndMillis !== undefined ? input.fallEndMillis : current.fall_end_millis,
            })

            await saveRecording(idRecording, nextRecording)

            const updated = await getRecordingById(idRecording, isTruthyFlag(req.query.includePoints))
            res.json(updated)
        } catch (err) {
            handleError(res, err, "Failed to update recording")
        }
    }

    router.put("/:id", updateRecordingHandler)
    router.patch("/:id", updateRecordingHandler)

    router.post("/:id/points", async (req, res) => {
        try {
            const idRecording = parseId(req.params.id, "id")
            const current = await getRecordingRow(idRecording)

            if (!current) {
                return res.status(404).json({error: "Recording not found"})
            }

            if (current.status === "completed") {
                return res.status(409).json({error: "Cannot append sensor points to a completed recording"})
            }

            const points = normalizePointsPayload(req.body)
            const values = points.map((point) => [
                idRecording,
                point.wallTimeMillis,
                point.timeText,
                point.ax,
                point.ay,
                point.az,
                point.aNorm,
                point.gx,
                point.gy,
                point.gz,
                point.qw,
                point.qx,
                point.qy,
                point.qz,
                point.vax,
                point.vay,
                point.vaz,
            ])

            await db.query(
                `INSERT INTO heimdall_sensor_points
                 (id_recording, wall_time_millis, time_text, ax, ay, az, a_norm, gx, gy, gz, qw, qx, qy, qz, vax, vay,
                  vaz)
                 VALUES ?`,
                [values]
            )

            const bounds = await getRecordingBounds(idRecording)
            await db.execute(
                `UPDATE heimdall_recordings
                 SET started_at_millis = ?,
                     ended_at_millis   = ?
                 WHERE id_recording = ?`,
                [bounds.started_at_millis, bounds.ended_at_millis, idRecording]
            )

            const recording = await getRecordingById(idRecording)

            res.status(201).json({
                message: "Sensor points appended successfully",
                insertedCount: points.length,
                recording,
            })
        } catch (err) {
            handleError(res, err, "Failed to append sensor points")
        }
    })

    router.get("/:id/points", async (req, res) => {
        try {
            const idRecording = parseId(req.params.id, "id")
            const current = await getRecordingRow(idRecording)

            if (!current) {
                return res.status(404).json({error: "Recording not found"})
            }

            const [rows] = await db.execute(
                `SELECT *
                 FROM heimdall_sensor_points
                 WHERE id_recording = ?
                 ORDER BY wall_time_millis ASC, id_sensor_point ASC`,
                [idRecording]
            )

            res.json(rows.map(formatPoint))
        } catch (err) {
            handleError(res, err, "Failed to fetch sensor points")
        }
    })

    router.post("/:id/finalize", async (req, res) => {
        try {
            const idRecording = parseId(req.params.id, "id")
            const current = await getRecordingRow(idRecording)

            if (!current) {
                return res.status(404).json({error: "Recording not found"})
            }

            const input = normalizeRecordingInput(req.body || {})
            const bounds = await getRecordingBounds(idRecording)

            const nextRecording = validateRecordingState({
                status: "completed",
                startedAtMillis: input.startedAtMillis !== undefined
                    ? input.startedAtMillis
                    : current.started_at_millis ?? bounds.started_at_millis,
                endedAtMillis: input.endedAtMillis !== undefined
                    ? input.endedAtMillis
                    : current.ended_at_millis ?? bounds.ended_at_millis,
                isFall: input.isFall !== undefined ? input.isFall : Boolean(current.is_fall),
                fallStartMillis: input.fallStartMillis !== undefined ? input.fallStartMillis : current.fall_start_millis,
                fallEndMillis: input.fallEndMillis !== undefined ? input.fallEndMillis : current.fall_end_millis,
            })

            await saveRecording(idRecording, nextRecording)

            const recording = await getRecordingById(idRecording, true)
            res.json(recording)
        } catch (err) {
            handleError(res, err, "Failed to finalize recording")
        }
    })

    router.delete("/:id", async (req, res) => {
        try {
            const idRecording = parseId(req.params.id, "id")
            const current = await getRecordingRow(idRecording)

            if (!current) {
                return res.status(404).json({error: "Recording not found"})
            }

            await db.execute("DELETE FROM heimdall_recordings WHERE id_recording = ?", [idRecording])
            res.json({message: "Recording deleted successfully", idRecording})
        } catch (err) {
            handleError(res, err, "Failed to delete recording")
        }
    })

    return router
}
