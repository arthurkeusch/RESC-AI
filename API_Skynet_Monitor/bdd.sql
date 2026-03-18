CREATE TABLE IF NOT EXISTS models
(
    id_model INT AUTO_INCREMENT,
    name     VARCHAR(255) UNIQUE,
    params   TEXT,
    filename VARCHAR(255) UNIQUE,
    size     BIGINT,
    PRIMARY KEY (id_model)
);

CREATE TABLE IF NOT EXISTS datasets
(
    id_datatset       INT AUTO_INCREMENT,
    name              TEXT,
    description       TEXT,
    is_conversational BOOLEAN,
    PRIMARY KEY (id_datatset)
);

CREATE TABLE IF NOT EXISTS prompts
(
    id_prompt   INT AUTO_INCREMENT,
    prompt      TEXT,
    id_datatset INT NOT NULL,
    PRIMARY KEY (id_prompt),
    FOREIGN KEY (id_datatset) REFERENCES datasets (id_datatset)
);

CREATE TABLE IF NOT EXISTS devices
(
    id_devices INT AUTO_INCREMENT,
    name       TEXT,
    PRIMARY KEY (id_devices)
);

CREATE TABLE IF NOT EXISTS prompts_results
(
    id_result  INT AUTO_INCREMENT,
    response   TEXT,
    id_prompt  INT NOT NULL,
    id_model   INT NOT NULL,
    id_devices INT NOT NULL,
    PRIMARY KEY (id_result),
    FOREIGN KEY (id_prompt) REFERENCES prompts (id_prompt),
    FOREIGN KEY (id_model) REFERENCES models (id_model),
    FOREIGN KEY (id_devices) REFERENCES devices (id_devices)
);

CREATE TABLE IF NOT EXISTS heimdall_recordings
(
    id_recording      BIGINT AUTO_INCREMENT,
    started_at_millis BIGINT      NULL,
    ended_at_millis   BIGINT      NULL,
    status            VARCHAR(32) NOT NULL DEFAULT 'recording',
    is_fall           BOOLEAN     NOT NULL DEFAULT FALSE,
    fall_start_millis BIGINT      NULL,
    fall_end_millis   BIGINT      NULL,
    created_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    PRIMARY KEY (id_recording),
    INDEX idx_heimdall_recordings_status (status),
    INDEX idx_heimdall_recordings_started_at (started_at_millis)
);

CREATE TABLE IF NOT EXISTS heimdall_sensor_points
(
    id_sensor_point  BIGINT AUTO_INCREMENT,

    id_recording     BIGINT      NOT NULL,
    wall_time_millis BIGINT      NOT NULL,
    time_text        VARCHAR(32) NULL,
    ax               FLOAT       NULL,
    ay               FLOAT       NULL,
    az               FLOAT       NULL,
    a_norm           FLOAT       NULL,
    gx               FLOAT       NULL,
    gy               FLOAT       NULL,
    gz               FLOAT       NULL,
    qw               FLOAT       NULL,
    qx               FLOAT       NULL,
    qy               FLOAT       NULL,
    qz               FLOAT       NULL,
    vax              FLOAT       NULL,
    vay              FLOAT       NULL,
    vaz              FLOAT       NULL,
    created_at       TIMESTAMP   NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (id_sensor_point),
    FOREIGN KEY (id_recording) REFERENCES heimdall_recordings (id_recording) ON DELETE CASCADE,
    INDEX idx_heimdall_sensor_points_recording_time (id_recording, wall_time_millis),
    INDEX idx_heimdall_sensor_points_wall_time (wall_time_millis)
);
