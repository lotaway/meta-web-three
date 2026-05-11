package com.metawebthree.common.cloud;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * S3 兼容对象存储服务
 * 封装 DefaultS3Service 实现 StorageService 接口
 */
@Slf4j
@Service
@ConditionalOnProperty(name = "storage.type", havingValue = "s3")
@RequiredArgsConstructor
public class S3StorageService implements StorageService {

    private final DefaultS3Service s3Service;
    private final DefaultS3Config s3Config;

    @Override
    public String uploadFile(MultipartFile file) {
        return uploadFile(file, file.getOriginalFilename());
    }

    @Override
    public String uploadFile(MultipartFile file, String fileName) {
        try {
            // 使用日期作为前缀
            SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd");
            String datePath = sdf.format(new Date());
            String key = datePath + "/" + System.currentTimeMillis() + "_" + fileName;
            
            s3Service.putObject(s3Config.getName(), key, file.getBytes());
            
            log.info("文件上传成功: {}", key);
            
            return getFileUrl(key);
        } catch (IOException e) {
            log.error("文件上传失败: {}", e.getMessage(), e);
            throw new RuntimeException("文件上传失败: " + e.getMessage(), e);
        }
    }

    @Override
    public void deleteFile(String objectName) {
        s3Service.deleteObject(s3Config.getName(), objectName);
    }

    @Override
    public String getFileUrl(String objectName) {
        return s3Service.getFileUrlWithCheck(s3Config.getName(), objectName).orElse(null);
    }

    @Override
    public byte[] getFile(String objectName) {
        return s3Service.getObject(s3Config.getName(), objectName).orElse(null);
    }

    @Override
    public boolean fileExists(String objectName) {
        return s3Service.getObject(s3Config.getName(), objectName).isPresent();
    }
}