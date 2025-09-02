package com.metawebthree.common.utils;

import com.fasterxml.jackson.databind.ObjectMapper;

import jakarta.annotation.PostConstruct;

import org.rocksdb.Options;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;
import org.rocksdb.RocksIterator;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

import java.io.File;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.StringJoiner;

@EnableScheduling
@Component
public class RocksDBManager {

    @Value("${logger.data-path:logs/rocksdb-data}")
    private String BASE_PATH;

    @Value("${logger.expire-time}")
    private long EXPIRE_TIME;

    static {
        RocksDB.loadLibrary();
    }

    private RocksDB db;
    private final ObjectMapper mapper = new ObjectMapper();

    @PostConstruct
    public void init() throws RocksDBException {
        boolean result = createDirectory(BASE_PATH);
        if (!result) {
            throw new RocksDBException("Failed to create directory: " + BASE_PATH);
        }
        String basePath = new File(BASE_PATH).getAbsolutePath();
        Options options = new Options().setCreateIfMissing(true);
        this.db = RocksDB.open(options, basePath);
    }

    public void init(String path, long expireTime) throws RocksDBException {
        this.close();
        this.BASE_PATH = path;
        this.EXPIRE_TIME = expireTime;
        this.init();
    }

    private boolean createDirectory(String path) {
        File file = new File(path);
        if (!file.exists()) {
            return file.mkdirs();
        }
        return true;
    }

    public void saveLog(String topic, String type, Object log) throws Exception {
        String key = getKey(topic, type);
        db.put(key.getBytes(), mapper.writeValueAsBytes(log));
    }

    protected StringJoiner getPrefix(String topic, String type) {
        var sj = new StringJoiner(":");
        sj.add("Business").add(topic).add(type);
        return sj;
    }

    protected String getKey(String topic, String type) {
        var sj = getPrefix(topic, type);
        sj.add(String.valueOf(System.currentTimeMillis()));
        String key = sj.toString();
        return key;
    }

    public <T> List<T> getLogs(String topic, String type, Class<T> clazz) throws Exception {
        List<T> result = new ArrayList<>();
        String prefix = getPrefix(topic, type).toString();
        RocksIterator it = db.newIterator();
        for (it.seek(prefix.getBytes()); it.isValid(); it.next()) {
            String k = new String(it.key());
            if (!k.startsWith(prefix))
                break;
            result.add(mapper.readValue(it.value(), clazz));
        }
        return result;
    }

    @Scheduled(cron = "0 0 3 * * ? ")
    public void cleanExpired() throws RocksDBException {
        RocksIterator it = db.newIterator();
        long now = Instant.now().toEpochMilli();
        for (it.seekToFirst(); it.isValid(); it.next()) {
            String key = new String(it.key());
            String[] parts = key.split(":");
            if (parts.length < 4)
                continue;
            long ts = Long.parseLong(parts[parts.length - 1]);
            if (now - ts > EXPIRE_TIME) {
                db.delete(it.key());
            }
        }
    }

    public void clean(String topic, String type) throws RocksDBException {
        String prefix = getPrefix(topic, type).toString();
        RocksIterator it = db.newIterator();
        for (it.seek(prefix.getBytes()); it.isValid(); it.next()) {
            String k = new String(it.key());
            if (!k.startsWith(prefix))
                break;
            db.delete(it.key());
        }
    }

    public void close() {
        if (db != null)
            db.close();
    }
}
