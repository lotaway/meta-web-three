package com.metawebthree.product;

import com.metawebthree.common.cloud.DefaultS3Buckets;
import com.metawebthree.common.cloud.DefaultS3Service;
import com.metawebthree.common.utils.RocketMQ.MQProducer;
import com.metawebthree.image.ProductImageService;
import org.apache.rocketmq.client.exception.MQBrokerException;
import org.apache.rocketmq.client.exception.MQClientException;
import org.apache.rocketmq.remoting.exception.RemotingException;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class ProductService {
    private final DefaultS3Service s3Service;
    private final DefaultS3Buckets s3Bucket;
    private final MQProducer mqProducer;

    private final ProductImageService productImageService;

    public ProductService(DefaultS3Service s3Service, DefaultS3Buckets s3Bucket, MQProducer mqProducer,
            ProductImageService productImageService) {
        this.s3Service = s3Service;
        this.s3Bucket = s3Bucket;
        this.mqProducer = mqProducer;
        this.productImageService = productImageService;
    }

    public PutObjectResponse createProduct(String key, byte[] content) {
        return s3Service.putObject(s3Bucket.getName(), key, content);
    }

    public PutObjectResponse updateProduct(String key, byte[] content) {
        return s3Service.putObject(s3Bucket.getName(), key, content);
    }

    public byte[] getProduct(String key) throws RuntimeException {
        return s3Service.getObject(s3Bucket.getName(), key);
    }

    public void deleteProduct(String key)
            throws MQBrokerException, RemotingException, InterruptedException, MQClientException {
        s3Service.deleteObject(s3Bucket.getName(), key);
        mqProducer.send("deleteProduct", "delete product with:" + key, null, null);
    }

    public void testNIOUpdateImage(String path) throws IOException {
        FileInputStream fs = new FileInputStream(new File(path));
        FileChannel fc = fs.getChannel();
        ByteBuffer buf = ByteBuffer.allocate(48);
        int result;
        while ((result = fc.read(buf)) != 0) {
            System.out.println(result);
        }
        fs.close();
        fc.close();
    }

    public boolean uploadImage(Long productId, MultipartFile file) {
        String imageId = UUID.randomUUID().toString();
        String url = "/product/%s/images/%s".formatted(productId, imageId);
        try {
            PutObjectResponse res = s3Service.putObject(s3Bucket.getName(), url, file.getBytes());
            System.out.println("Image uploaded successfully: " + res);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return Integer.valueOf(1).equals(productImageService.create(productId, imageId, url));
    }

    public byte[] getImages(Long productId) {
        return s3Service.getObject(s3Bucket.getName(), "/product/%s/images/".formatted(productId));
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
