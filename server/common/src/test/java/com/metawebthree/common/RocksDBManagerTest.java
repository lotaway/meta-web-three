package com.metawebthree.common;

import java.util.List;

import org.junit.Assert;
import org.junit.Test;

import com.metawebthree.common.utils.RocksDBManager;

public class RocksDBManagerTest {

    private RocksDBManager rocksDBManager = new RocksDBManager();

    @Test
    public void testCanWriteAndRead() throws Exception {
        // reset to whole project directory
        rocksDBManager.init("../../logs/rocksdb-test", 60 * 1000);
        var productLog = new ProductLog();
        productLog.productId = "12345";
        productLog.quantity = 10;
        String topic = "product";
        String type = "create";
        
        rocksDBManager.saveLog(topic, type, productLog);
        List<ProductLog> result = rocksDBManager.getLogs(topic, type, ProductLog.class);
        Assert.assertTrue(result.size() > 0);
        var log = result.get(0);
        Assert.assertEquals(productLog.productId, log.productId);
        Assert.assertEquals(productLog.quantity, log.quantity);
        
        rocksDBManager.clean(topic, type);
        List<ProductLog> clearResult = rocksDBManager.getLogs(topic, type, ProductLog.class);
        Assert.assertEquals(0, clearResult.size());
        rocksDBManager.close();
    }

    public static class ProductLog {
        public String productId;
        public int quantity;
    }
}
