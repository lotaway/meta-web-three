package com.metawebthree.common.cloud;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

/**
 * MinIO 对象存储服务
 * 封装 MinioService 实现 StorageService 接口
 */
@Slf4j
@Service
@ConditionalOnProperty(name = "storage.type", havingValue = "minio")
@RequiredArgsConstructor
public class MinioStorageService implements StorageService {

    private final MinioService minioService;

    @Override
    public String uploadFile(MultipartFile file) {
        return minioService.uploadFile(file);
    }

    @Override
    public String uploadFile(MultipartFile file, String fileName) {
        // MinioService 的 uploadFile 方法已经包含文件名，这里直接复用
        return minioService.uploadFile(file);
    }

    @Override
    public void deleteFile(String objectName) {
        minioService.deleteFile(objectName);
    }

    @Override
    public String getFileUrl(String objectName) {
        return minioService.getFileUrl(objectName);
    }

    @Override
    public byte[] getFile(String objectName) {
        return minioService.getFile(objectName);
    }

    @Override
    public boolean fileExists(String objectName) {
        return minioService.fileExists(objectName);
    }
}