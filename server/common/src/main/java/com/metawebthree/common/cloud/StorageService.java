package com.metawebthree.common.cloud;

import org.springframework.web.multipart.MultipartFile;


/**
 * 存储服务抽象接口
 * 支持本地存储和 MinIO 等 S3 兼容存储
 */
public interface StorageService {

    /**
     * 上传文件
     * @param file 文件
     * @return 存储后的文件访问URL
     */
    String uploadFile(MultipartFile file);

    /**
     * 上传文件并指定文件名
     * @param file 文件
     * @param fileName 指定的文件名
     * @return 存储后的文件访问URL
     */
    String uploadFile(MultipartFile file, String fileName);

    /**
     * 删除文件
     * @param objectName 文件key/路径
     */
    void deleteFile(String objectName);

    /**
     * 获取文件URL
     * @param objectName 文件key/路径
     * @return 文件访问URL
     */
    String getFileUrl(String objectName);

    /**
     * 获取文件内容
     * @param objectName 文件key/路径
     * @return 文件内容
     */
    byte[] getFile(String objectName);

    /**
     * 检查文件是否存在
     * @param objectName 文件key/路径
     * @return 是否存在
     */
    boolean fileExists(String objectName);

    /**
     * 存储类型枚举
     */
    enum StorageType {
        LOCAL,   // 本地文件系统
        MINIO,   // MinIO 对象存储
        S3       // AWS S3 或其他 S3 兼容存储
    }
}