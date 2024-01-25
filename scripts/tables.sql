
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
CREATE INDEX IF NOT EXISTS exec_name ON execs (name);-- 

CREATE TABLE IF NOT EXISTS packs (
	_id SERIAL NOT NULL, 
	exec_id INTEGER NOT NULL, 
	created_time TIMESTAMP WITHOUT TIME ZONE, 
	name VARCHAR(256), 
	tag VARCHAR(256), 
	config JSON, 
	command JSON, 
	PRIMARY KEY (_id), 
	FOREIGN KEY(exec_id) REFERENCES execs (_id)
)

;-- 
CREATE INDEX IF NOT EXISTS pack_query ON packs (name, exec_id);-- 
CREATE INDEX IF NOT EXISTS pack_tag ON packs (tag);-- 
CREATE INDEX IF NOT EXISTS exec_pack_query ON packs (exec_id);-- 

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
