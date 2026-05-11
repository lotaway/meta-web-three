package com.metawebthree.common.cloud;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.Date;

/**
 * 本地文件系统存储服务
 * 复用现有的 /media/upload/file 逻辑
 */
@Slf4j
@Service
@ConditionalOnProperty(name = "storage.type", havingValue = "local", matchIfMissing = true)
public class LocalStorageService implements StorageService {

    @Value("${upload.path:/upload/file/}")
    private String uploadPath;

    @Value("${upload.base-url:}")
    private String baseUrl;

    @Override
    public String uploadFile(MultipartFile file) {
        return uploadFile(file, file.getOriginalFilename());
    }

    @Override
    public String uploadFile(MultipartFile file, String fileName) {
        try {
            // 使用日期作为子目录
            SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd");
            String datePath = sdf.format(new Date());
            
            // 构建完整的文件路径
            String objectName = datePath + "/" + System.currentTimeMillis() + "_" + fileName;
            File destFile = new File(uploadPath + objectName);
            
            // 确保父目录存在
            File parentDir = destFile.getParentFile();
            if (!parentDir.exists()) {
                parentDir.mkdirs();
            }
            
            // 写入文件
            FileUtils.writeByteArrayToFile(destFile, file.getBytes());
            
            log.info("文件上传成功: {}", objectName);
            
            // 返回文件访问URL
            return getFileUrl(objectName);
        } catch (IOException e) {
            log.error("文件上传失败: {}", e.getMessage(), e);
            throw new RuntimeException("文件上传失败: " + e.getMessage(), e);
        }
    }

    @Override
    public void deleteFile(String objectName) {
        try {
            File file = new File(uploadPath + objectName);
            if (file.exists()) {
                file.delete();
                log.info("文件删除成功: {}", objectName);
            } else {
                log.warn("文件不存在: {}", objectName);
            }
        } catch (Exception e) {
            log.error("文件删除失败: {}", e.getMessage(), e);
            throw new RuntimeException("文件删除失败: " + e.getMessage(), e);
        }
    }

    @Override
    public String getFileUrl(String objectName) {
        // 如果配置了 baseUrl，使用 baseUrl
        if (baseUrl != null && !baseUrl.isEmpty()) {
            return baseUrl + "/" + objectName;
        }
        // 否则返回相对路径
        return "/upload/file/" + objectName;
    }

    @Override
    public byte[] getFile(String objectName) {
        try {
            File file = new File(uploadPath + objectName);
            if (file.exists()) {
                return FileUtils.readFileToByteArray(file);
            }
            return null;
        } catch (IOException e) {
            log.error("读取文件失败: {}", e.getMessage(), e);
            throw new RuntimeException("读取文件失败: " + e.getMessage(), e);
        }
    }

    @Override
    public boolean fileExists(String objectName) {
        File file = new File(uploadPath + objectName);
        return file.exists();
    }
}