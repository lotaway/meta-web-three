package com.metawebthree.common.cloud;

import io.minio.BucketExistsArgs;
import io.minio.MakeBucketArgs;
import io.minio.MinioClient;
import io.minio.PutObjectArgs;
import io.minio.RemoveObjectArgs;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Optional;

/**
 * MinIO对象存储服务
 * 参考项目: mall-admin/src/main/java/com/macro/mall/controller/MinioController.java
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MinioService {

    private final MinioClient minioClient;
    private final MinioConfig minioConfig;

    /**
     * 上传文件到MinIO
     */
    public String uploadFile(MultipartFile file) {
        try {
            String filename = file.getOriginalFilename();
            SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd");
            String objectName = sdf.format(new Date()) + "/" + System.currentTimeMillis() + "_" + filename;

            // 检查存储桶是否存在
            boolean exists = minioClient.bucketExists(
                    BucketExistsArgs.builder().bucket(minioConfig.getBucketName()).build());
            if (!exists) {
                log.info("Creating bucket: {}", minioConfig.getBucketName());
                minioClient.makeBucket(
                        MakeBucketArgs.builder().bucket(minioConfig.getBucketName()).build());
            }

            // 上传文件
            InputStream inputStream = file.getInputStream();
            minioClient.putObject(
                    PutObjectArgs.builder()
                            .bucket(minioConfig.getBucketName())
                            .object(objectName)
                            .contentType(file.getContentType())
                            .stream(inputStream, file.getSize(), -1)
                            .build());

            log.info("File uploaded successfully: {}", objectName);
            
            // 返回文件URL
            return getFileUrl(objectName);
        } catch (Exception e) {
            log.error("Failed to upload file to MinIO", e);
            throw new RuntimeException("Failed to upload file to MinIO: " + e.getMessage(), e);
        }
    }

    /**
     * 删除MinIO中的文件
     */
    public void deleteFile(String objectName) {
        try {
            // 从完整URL中提取objectName
            String key = extractObjectKey(objectName);
            
            minioClient.removeObject(
                    RemoveObjectArgs.builder()
                            .bucket(minioConfig.getBucketName())
                            .object(key)
                            .build());
            
            log.info("File deleted successfully: {}", key);
        } catch (Exception e) {
            log.error("Failed to delete file from MinIO", e);
            throw new RuntimeException("Failed to delete file from MinIO: " + e.getMessage(), e);
        }
    }

    /**
     * 获取文件URL
     */
    public String getFileUrl(String objectName) {
        String key = extractObjectKey(objectName);
        return minioConfig.getEndpoint() + "/" + minioConfig.getBucketName() + "/" + key;
    }

    /**
     * 从URL或key中提取objectKey
     */
    private String extractObjectKey(String objectName) {
        if (objectName.contains("/" + minioConfig.getBucketName() + "/")) {
            // 是完整URL，提取bucket后面的部分
            return objectName.substring(objectName.indexOf(minioConfig.getBucketName()) + minioConfig.getBucketName().length() + 1);
        }
        return objectName;
    }

    /**
     * 检查文件是否存在
     */
    public boolean fileExists(String objectName) {
        try {
            String key = extractObjectKey(objectName);
            minioClient.statObject(
                    io.minio.StatObjectArgs.builder()
                            .bucket(minioConfig.getBucketName())
                            .object(key)
                            .build());
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}