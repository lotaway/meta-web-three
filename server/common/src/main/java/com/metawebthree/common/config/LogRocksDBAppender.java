package com.metawebthree.common.config;

import ch.qos.logback.core.AppenderBase;
import ch.qos.logback.classic.spi.ILoggingEvent;
import org.rocksdb.Options;
import org.rocksdb.RocksDB;
import org.rocksdb.RocksDBException;

public class LogRocksDBAppender extends AppenderBase<ILoggingEvent> {
    private RocksDB db;
    private static long seq = 0;

    @Override
    public void start() {
        try {
            RocksDB.loadLibrary();
            try (Options options = new Options().setCreateIfMissing(true)) {
                db = RocksDB.open(options, "/tmp/logs-rocksdb");
                super.start();
            }
        } catch (RocksDBException e) {
            addError("Failed to open RocksDB", e);
        }
    }

    @Override
    protected void append(ILoggingEvent eventObject) {
        try {
            String message = eventObject.getFormattedMessage();
            String level = eventObject.getLevel().toString();
            String thread = eventObject.getThreadName();
            String log = String.format("[%s] [%s] %s", level, thread, message);

            byte[] key = String.valueOf(seq++).getBytes();
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
