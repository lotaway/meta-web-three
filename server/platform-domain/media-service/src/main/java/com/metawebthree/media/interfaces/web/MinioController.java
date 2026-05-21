package com.metawebthree.media.interfaces.web;

import com.metawebthree.common.cloud.MinioService;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

/**
 * MinIO对象存储管理Controller
 * 参考项目: mall-admin/src/main/java/com/macro/mall/controller/MinioController.java
 */
@Slf4j
@RestController
@RequestMapping("/minio")
@Tag(name = "MinioController", description = "MinIO对象存储管理")
@RequiredArgsConstructor
public class MinioController {

    private final MinioService minioService;

    @Operation(summary = "文件上传", description = "上传文件到MinIO对象存储")
    @PostMapping("/upload")
    public ApiResponse<String> upload(@RequestPart("file") MultipartFile file) {
        try {
            if (file.isEmpty()) {
                return ApiResponse.error(ResponseStatus.PARAM_ERROR, "文件不能为空");
            }
            String url = minioService.uploadFile(file);
            return ApiResponse.success(url);
        } catch (Exception e) {
            log.error("文件上传失败: {}", e.getMessage(), e);
            return ApiResponse.error(ResponseStatus.MEDIA_UPLOAD_FAILED, e.getMessage());
        }
    }

    @Operation(summary = "文件删除", description = "从MinIO对象存储中删除文件")
    @PostMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam("objectName") String objectName) {
        try {
            minioService.deleteFile(objectName);
            return ApiResponse.success();
        } catch (Exception e) {
            log.error("文件删除失败: {}", e.getMessage(), e);
            return ApiResponse.error(ResponseStatus.MEDIA_DELETE_FAILED, e.getMessage());
        }
    }

    @Operation(summary = "获取文件URL", description = "获取MinIO中文件的访问URL")
    @GetMapping("/url")
    public ApiResponse<String> getUrl(@RequestParam("objectName") String objectName) {
        try {
            String url = minioService.getFileUrl(objectName);
            return ApiResponse.success(url);
        } catch (Exception e) {
            log.error("获取文件URL失败: {}", e.getMessage(), e);
            return ApiResponse.error(ResponseStatus.MEDIA_NOT_FOUND, e.getMessage());
        }
    }

    @Operation(summary = "检查文件是否存在", description = "检查MinIO中是否存在指定文件")
    @GetMapping("/exists")
    public ApiResponse<Boolean> exists(@RequestParam("objectName") String objectName) {
        try {
            boolean exists = minioService.fileExists(objectName);
            return ApiResponse.success(exists);
        } catch (Exception e) {
            log.error("检查文件是否存在失败: {}", e.getMessage(), e);
            return ApiResponse.error(ResponseStatus.MEDIA_NOT_FOUND, e.getMessage());
        }
    }
}