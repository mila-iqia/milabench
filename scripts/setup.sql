-- 

CREATE TABLE execs (
	_id SERIAL NOT NULL, 
	name VARCHAR(256), 
	namespace VARCHAR(256), 
	created_time TIMESTAMP WITHOUT TIME ZONE, 
	meta JSON, 
	status VARCHAR(256), 
	PRIMARY KEY (_id)
);

-- 
-- 

CREATE TABLE packs (
	_id SERIAL NOT NULL, 
	exec_id INTEGER NOT NULL, 
	created_time TIMESTAMP WITHOUT TIME ZONE, 
	name VARCHAR(256), 
	tag VARCHAR(256), 
	config JSON, 
	PRIMARY KEY (_id), 
	FOREIGN KEY(exec_id) REFERENCES execs (_id)
);

-- 
-- 

CREATE TABLE metrics (
	_id SERIAL NOT NULL, 
	exec_id INTEGER NOT NULL, 
	pack_id INTEGER NOT NULL, 
	name VARCHAR(256), 
	value FLOAT, 
	gpu_id INTEGER, 
	PRIMARY KEY (_id), 
	FOREIGN KEY(exec_id) REFERENCES execs (_id), 
	FOREIGN KEY(pack_id) REFERENCES packs (_id)
);

-- 
