package com.metawebthree.media.interfaces.web;
import com.metawebthree.media.application.*;
import com.metawebthree.media.application.dto.*;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.common.enums.ResponseStatus;

import io.swagger.v3.oas.annotations.tags.Tag;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;


@RestController
@RequestMapping("/media")
@Tag(name = "Media Management")
public class MediaController {

    private final MediaService mediaService;

    @Value("${upload.path:/upload/file/}")
    private String uploadPath;

    public MediaController(MediaService mediaService) {
        this.mediaService = mediaService;
    }

    @PostMapping
    public String uploadMedia(@RequestParam("file") MultipartFile file) {
        return mediaService.uploadFile(file);
    }

    @PostMapping("/upload/file")
    public ApiResponse<Void> file(@RequestParam("file") MultipartFile file) {
        String fileName = file.getOriginalFilename();
        try {
            File destFile = new File(uploadPath + fileName);
            FileUtils.writeByteArrayToFile(destFile, file.getBytes());
        } catch (IOException e) {
            return ApiResponse.error(ResponseStatus.MEDIA_UPLOAD_FAILED);
        }
        return ApiResponse.success();
    }

    @GetMapping("/{key}")
    public byte[] getMedia(@PathVariable String key) {
        return mediaService.getMedia(key).orElse(new byte[]{});
    }

    @DeleteMapping("/{key}")
    public void deleteMedia(@PathVariable String key) {
        mediaService.deleteMedia(key);
    }
}

    @PostMapping
    public String uploadMedia(@RequestParam("file") MultipartFile file) {
        return mediaService.uploadFile(file);
    }

    @PostMapping("/upload/file")
    public ApiResponse<Void> file(@RequestParam("file") MultipartFile file) {
        String fileName = file.getOriginalFilename();
        try {
            // TODO: Avoid hardcoded paths
            File destFile = new File("/upload/file/" + fileName);
            FileUtils.writeByteArrayToFile(destFile, file.getBytes());
        } catch (IOException e) {
            return ApiResponse.error(e.getMessage());
        }
        return ApiResponse.success();
    }
    

    @GetMapping("/{key}")
    public byte[] getMedia(@PathVariable String key) {
        return mediaService.getMedia(key).orElse(new byte[]{});
    }

    @DeleteMapping("/{key}")
    public void deleteMedia(@PathVariable String key) {
        mediaService.deleteMedia(key);
    }
}
