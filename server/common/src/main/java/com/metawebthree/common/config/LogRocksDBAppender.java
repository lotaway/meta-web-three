package com.metawebthree.common.config;

import ch.qos.logback.core.AppenderBase;
import ch.qos.logback.classic.spi.ILoggingEvent;

import java.io.File;
import java.util.Properties;
import java.util.StringJoiner;

import org.rocksdb.Options;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;

public class LogRocksDBAppender extends AppenderBase<ILoggingEvent> {
    private RocksDB db;
    private static long seq = 0;
    private static final String SEQ_KEY = "SystemOut";
    // @TODO diff micro service can't use same path, it's lock
    private String LOG_PATH = "logs/rocksdb-logs";

    @Override
    public void start() {
        // var supportUtil = new JavaUtil();
        // String instanceId = supportUtil.getInstanceId();
        Properties properties = System.getProperties();
        System.out.println("properties: " + properties);
        String serviceName = System.getProperty("APPLICATION_NAME");
        if (serviceName == null) {
            serviceName = System.getProperty("spring.boot.project.name", "default");
        }
        RocksDB.loadLibrary();
        try (final Options options = new Options().setCreateIfMissing(true)) {
            var sj = new StringJoiner("/");
            sj.add(LOG_PATH).add(serviceName);
            // sj.add(instanceId);
            String path = new File(sj.toString()).getAbsolutePath();
            boolean result = createDirectory(path);
            if (!result) {
                System.out.println("Warning: Log directory created failed.");
            }
            db = RocksDB.open(options, path);
            byte[] preSeq = db.get(SEQ_KEY.getBytes());
            if (preSeq != null && preSeq.length > 0) {
                try {
                    seq = Long.parseLong(new String(preSeq));
                } catch (NumberFormatException e) {
                    // do nothing
                }
            }
            super.start();
        } catch (RocksDBException e) {
            addError("Failed to open RocksDB", e);
        }
    }

    private boolean createDirectory(String path) {
        File file = new File(path);
        if (!file.exists()) {
            return file.mkdirs();
        }
        return true;
    }

    @Override
    protected void append(ILoggingEvent eventObject) {
        try {
            String message = eventObject.getFormattedMessage();
            String level = eventObject.getLevel().toString();
            String thread = eventObject.getThreadName();
            String log = String.format("[%s] [%s] %s", level, thread, message);
            byte[] seqKey = SEQ_KEY.getBytes();
            db.put(seqKey, String.valueOf(seq).getBytes());
            byte[] key = String.format("%s:%d", SEQ_KEY, seq++).getBytes();
            db.put(key, log.getBytes());
        } catch (Exception e) {
            addError("Error writing log to RocksDB", e);
        }
    }

    @Override
    public void stop() {
        if (db != null) {
            db.close();
        }
        super.stop();
    }
}
