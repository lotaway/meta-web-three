package com.metawebthree.media.interfaces.web;

import com.metawebthree.common.cloud.StorageService;
import com.metawebthree.common.constants.HeaderConstants;
import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;
import com.metawebthree.media.application.*;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/media")
@Tag(name = "Media Management")
public class MediaController {

    private final MediaService mediaService;
    private final StorageService storageService;
    private final UploadQuotaService uploadQuotaService;

    public MediaController(MediaService mediaService, StorageService storageService, UploadQuotaService uploadQuotaService) {
        this.mediaService = mediaService;
        this.storageService = storageService;
        this.uploadQuotaService = uploadQuotaService;
    }

    @Operation(summary = "文件上传", description = "使用配置的存储服务上传文件(支持本地存储/MinIO/S3)")
    @PostMapping
    public ApiResponse<String> uploadMedia(
            @RequestParam("file") MultipartFile file,
            @RequestHeader(HeaderConstants.USER_ROLE) String userRole,
            @RequestHeader(value = HeaderConstants.USER_ID, required = false) Long userId) {
        try {
            uploadQuotaService.checkQuota(userRole, file.getSize(), userId);
            String url = storageService.uploadFile(file);
            uploadQuotaService.trackUpload(userId, file.getSize());
            return ApiResponse.success(url);
        } catch (IllegalArgumentException e) {
            return ApiResponse.error(ResponseStatus.MEDIA_UPLOAD_FAILED, e.getMessage());
        } catch (Exception e) {
            return ApiResponse.error(ResponseStatus.MEDIA_UPLOAD_FAILED, e.getMessage());
        }
    }

    @Operation(summary = "文件删除", description = "使用配置的存储服务删除文件")
    @DeleteMapping("/{key}")
    public ApiResponse<Void> deleteMedia(@PathVariable String key) {
        try {
            storageService.deleteFile(key);
            return ApiResponse.success();
        } catch (Exception e) {
            return ApiResponse.error(ResponseStatus.MEDIA_DELETE_FAILED, e.getMessage());
        }
    }

    @Operation(summary = "获取文件", description = "获取文件内容")
    @GetMapping("/{key}")
    public byte[] getMedia(@PathVariable String key) {
        return mediaService.getMedia(key).orElse(new byte[] {});
    }

    @Operation(summary = "获取文件访问URL", description = "获取文件的访问URL")
    @GetMapping("/url/{key}")
    public ApiResponse<String> getMediaUrl(@PathVariable String key) {
        String url = storageService.getFileUrl(key);
        return ApiResponse.success(url);
    }

    @Operation(summary = "检查文件是否存在", description = "检查存储服务中是否存在指定文件")
    @GetMapping("/exists/{key}")
    public ApiResponse<Boolean> exists(@PathVariable String key) {
        boolean exists = storageService.fileExists(key);
        return ApiResponse.success(exists);
    }
}