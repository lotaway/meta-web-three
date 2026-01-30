package com.metawebthree.media;

import com.metawebthree.common.dto.ApiResponse;

import io.swagger.v3.oas.annotations.tags.Tag;

import java.io.File;
import java.io.IOException;

import org.apache.commons.io.FileUtils;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;


@RestController
@RequestMapping("/media")
@Tag(name = "Media Management")
public class MediaController {

    private final MediaService mediaService;

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
