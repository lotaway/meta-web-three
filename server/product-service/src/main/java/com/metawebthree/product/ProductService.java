package com.metawebthree.product;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.common.utils.RocketMQ.MQProducer;
import com.metawebthree.image.ProductImageService;

import lombok.extern.slf4j.Slf4j;

import org.apache.rocketmq.client.exception.MQBrokerException;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.remoting.exception.RemotingException;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.util.concurrent.ConcurrentHashMap;

@Service
@Slf4j
public class ProductService {
    private final MQProducer mqProducer;
    private final ProductImageService productImageService;

    public ProductService(MQProducer mqProducer, ProductImageService productImageService) {
        this.mqProducer = mqProducer;
        this.productImageService = productImageService;
    }

    public Boolean createProduct() {
        Long id = IdWorker.getId();
        // @TODO sql
        return Boolean.valueOf(true);
    }

    public boolean updateProduct(Long id, byte[] description) {
        // @TODO sql modify
        return true;
    }

    public void deleteProduct(String key)
            throws MQBrokerException, RemotingException, InterruptedException, MQClientException {
        mqProducer.send("deleteProduct", "delete product with:" + key, null, null);
    }

    public boolean uploadImage(Long productId, MultipartFile imageFile) {
        String imageId = String.valueOf(IdWorker.getId());
        // @TODO upload image with MediaService
        return saveImage(productId, imageId);
    }

    public boolean saveImage(Long productId, String imageUrl) {
        String imageId = String.valueOf(IdWorker.getId());
        return Integer.valueOf(1).equals(productImageService.create(productId, imageId, imageUrl));
    }

    ConcurrentHashMap<Long, Object> statisticLockMap = new ConcurrentHashMap<>();
    Long count = 0L;

    public boolean updateFeatureStatistic(Long featureId, Integer increated) {
        return updateFeatureStatisticWithLock(featureId, increated);
    }

    public boolean updateFeatureStatisticWithLock(Long featureId, Integer increated) {
        Object lock = statisticLockMap.computeIfAbsent(featureId, key -> new Object());
        synchronized (lock) {
            count += increated;
        }
        return true;
    }
}
