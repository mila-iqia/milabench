--
-- Generated using:
--
--      python -m milabench.metrics.sqlalchemy
--

CREATE TABLE IF NOT EXISTS execs (
	_id SERIAL NOT NULL, 
	name VARCHAR(256), 
	namespace VARCHAR(256), 
	created_time TIMESTAMP WITHOUT TIME ZONE, 
	meta JSON, 
	status VARCHAR(256), 
	PRIMARY KEY (_id)
)

;-- 
CREATE INDEX IF NOT EXISTS execs_meta_gpus_0_product_idx ON execs USING btree ((meta -> 'accelerators' -> 'gpus' -> '0' ->> 'product'));-- 
CREATE INDEX IF NOT EXISTS idx_exec_meta_pytorch_torch ON execs USING btree ((meta -> 'pytorch' ->> 'torch'));-- 
-- CREATE INDEX IF NOT EXISTS idx_exec_meta_accelerators_gin ON execs USING gin (meta -> 'accelerators');-- 
CREATE INDEX IF NOT EXISTS exec_name ON execs (name);-- 
-- CREATE INDEX IF NOT EXISTS idx_exec_meta_pytorch_gin ON execs USING gin (meta -> 'pytorch');-- 
CREATE INDEX IF NOT EXISTS idx_exec_meta_pytorch_version ON execs USING btree ((meta -> 'pytorch' ->> 'version'));-- 

CREATE TABLE IF NOT EXISTS saved_queries (
	_id SERIAL NOT NULL, 
	name VARCHAR(256), 
	query JSON, 
	created_time TIMESTAMP WITHOUT TIME ZONE, 
	PRIMARY KEY (_id)
)

;-- 
CREATE INDEX IF NOT EXISTS saved_queries_name ON saved_queries (name);-- 

CREATE TABLE IF NOT EXISTS weights (
	_id SERIAL NOT NULL, 
	profile VARCHAR(256) NOT NULL, 
	pack VARCHAR(256) NOT NULL, 
	weight INTEGER NOT NULL, 
	priority INTEGER NOT NULL, 
	enabled BOOLEAN NOT NULL, 
	group1 VARCHAR(256), 
	group2 VARCHAR(256), 
	group3 VARCHAR(256), 
	group4 VARCHAR(256), 
	PRIMARY KEY (_id), 
	CONSTRAINT uq_profile_pack UNIQUE (profile, pack)
)

;-- 
CREATE INDEX IF NOT EXISTS weight_profile_pack ON weights (profile, pack);-- 
CREATE INDEX IF NOT EXISTS idx_weight_profile_priority ON weights (profile, priority);-- 
CREATE INDEX IF NOT EXISTS idx_weight_pack ON weights (pack);-- 
CREATE INDEX IF NOT EXISTS idx_weight_profile_enabled ON weights (profile, enabled);-- 

CREATE TABLE IF NOT EXISTS packs (
	_id SERIAL NOT NULL, 
	exec_id INTEGER NOT NULL, 
	created_time TIMESTAMP WITHOUT TIME ZONE, 
	name VARCHAR(256), 
	tag VARCHAR(256), 
	config JSON, 
	command JSON, 
	status VARCHAR(256), 
	ngpu INTEGER, 
	PRIMARY KEY (_id), 
	FOREIGN KEY(exec_id) REFERENCES execs (_id)
)

;-- 
CREATE INDEX IF NOT EXISTS exec_pack_query ON packs (exec_id);-- 
CREATE INDEX IF NOT EXISTS pack_query ON packs (name, exec_id);-- 
CREATE INDEX IF NOT EXISTS idx_pack_name ON packs (name);-- 
CREATE INDEX IF NOT EXISTS pack_tag ON packs (tag);-- 

CREATE TABLE IF NOT EXISTS metrics (
	_id SERIAL NOT NULL, 
	exec_id INTEGER NOT NULL, 
	pack_id INTEGER NOT NULL, 
	"order" INTEGER, 
	name VARCHAR(256), 
	namespace VARCHAR(256), 
	value FLOAT, 
	unit VARCHAR(128), 
	job_id INTEGER, 
	gpu_id VARCHAR(36), 
	PRIMARY KEY (_id), 
	FOREIGN KEY(exec_id) REFERENCES execs (_id), 
	FOREIGN KEY(pack_id) REFERENCES packs (_id)
)

;-- 
CREATE INDEX IF NOT EXISTS metric_query ON metrics (exec_id, pack_id);-- 
CREATE INDEX IF NOT EXISTS metric_name ON metrics (name);-- 
CREATE INDEX IF NOT EXISTS idx_metric_pack_name ON metrics (pack_id, name);-- 
CREATE INDEX IF NOT EXISTS idx_metric_exec_name ON metrics (exec_id, name);-- 
CREATE INDEX IF NOT EXISTS idx_metric_name_value ON metrics (name, value);-- 
CREATE INDEX IF NOT EXISTS idx_metric_exec_pack_name ON metrics (exec_id, pack_id, name);-- 

INSERT INTO
    weights (profile, weight, priority, pack, enabled, group1, group2)
VALUES
    ('default', 0, 1000, 'fp16', TRUE, 'SYNTHETIC', 'FLOPS'),
    ('default', 0, 1001, 'bf16', TRUE, 'SYNTHETIC', 'FLOPS'),
    ('default', 0, 1002, 'tf32', TRUE, 'SYNTHETIC', 'FLOPS'),
    ('default', 0, 1003, 'fp32', TRUE, 'SYNTHETIC', 'FLOPS'),
    ('default', 0, 2201, 'convnext_large-fp32', TRUE, 'CV', 'CONVNET'),
    ('default', 0, 2202, 'convnext_large-fp16', TRUE, 'CV', 'CONVNET'),
    ('default', 0, 2203, 'convnext_large-tf32', TRUE, 'CV', 'CONVNET'),
    ('default', 1, 2204, 'convnext_large-tf32-fp16', TRUE, 'CV', 'CONVNET'),
    ('default', 1, 2205, 'resnet50', TRUE, 'CV', 'CONVNET'),
    ('default', 0, 2206, 'resnet50-noio', TRUE, 'CV', 'CONVNET'),
    ('default', 0, 2207, 'resnet152-ddp-gpus', TRUE, 'CV', 'CONVNET'),
    ('default', 1, 2208, 'regnet_y_128gf', TRUE, 'CV', 'CONVNET'),
    ('default', 0, 2209, 'lightning', TRUE, 'CV', 'CONVNET'),
    ('default', 1, 2210, 'lightning-gpus', TRUE, 'CV', 'CONVNET'),
    ('default', 0, 2211, 'focalnet', TRUE, 'CV', 'CONVNET'),
    ('default', 0, 2012, 'diffusion-single', TRUE, 'CV', 'DIFFUSION'),
    ('default', 1, 2013, 'diffusion-gpus', TRUE, 'CV', 'DIFFUSION'),
    ('default', 1, 2014, 'diffusion-nodes', FALSE, 'CV', 'DIFFUSION'),
    ('default', 0, 2101, 'dinov2-giant-single', TRUE, 'CV', 'TRANSFORMER'),
    ('default', 1, 2102, 'dinov2-giant-gpus', TRUE, 'CV', 'TRANSFORMER'),
    ('default', 0, 2103, 'dinov2-giant-nodes', FALSE, 'CV', 'TRANSFORMER'),
    ('default', 1, 2104, 'llava-single', TRUE, 'CV', 'TRANSFORMER'),
    ('default', 0, 2105, 'llava-gpus', FALSE, 'CV', 'TRANSFORMER'),
    ('default', 1, 2106, 'vjepa-single', TRUE, 'CV', 'TRANSFORMER'),
    ('default', 1, 2107, 'vjepa-gpus', TRUE, 'CV', 'TRANSFORMER'),
    ('default', 0, 3100, 'bert-fp32', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 0, 3101, 'bert-fp16', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 0, 3102, 'bert-tf32', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3103, 'bert-tf32-fp16', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 0, 3104, 't5', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3105, 'reformer', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 0, 3106, 'whisper', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3107, 'llama', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3108, 'llm-lora-single', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3109, 'llm-lora-ddp-gpus', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3110, 'llm-lora-ddp-nodes', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3111, 'llm-lora-mp-gpus', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3112, 'llm-full-mp-gpus', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3113, 'llm-full-mp-nodes', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 3114, 'rlhf-single', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 0, 3115, 'rlhf-gpus', TRUE, 'NLP', 'TRANSFORMER'),
    ('default', 1, 4201, 'torchatari', TRUE, 'RL', 'CONVNET'),
    ('default', 1, 4302, 'brax', TRUE, 'RL', 'MLP'),
    ('default', 0, 4303, 'dqn', TRUE, 'RL', 'MLP'),
    ('default', 1, 4304, 'ppo', TRUE, 'RL', 'MLP'),
    ('default', 0, 4305, 'cleanrljax', FALSE, 'RL', 'MLP'),
    ('default', 1, 5000, 'pna', TRUE, 'GRAPHS', 'GNN'),
    ('default', 1, 5001, 'dimenet', TRUE, 'GRAPHS', 'GNN'),
    ('default', 1, 5002, 'recursiongfn', TRUE, 'GRAPHS', 'GFlow')
;-- 
