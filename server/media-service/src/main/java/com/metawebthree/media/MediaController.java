package com.metawebthree.media;

import com.metawebthree.common.cloud.DefaultS3Service;
import com.metawebthree.media.service.MediaService;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/media")
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
    public ApiResponse<Exception> file(MultipartFile file) {
        String fileName = file.getOriginalFilename();
        try {
            File destFile = new File("/upload/file/" + fileName);
            FileUtils.writeByteArrayToFile(destFile, file.getBytes());
        } catch (IOException e) {
            return ApiResponse.error(e);
        }
        return ApiResponse.success();
    }

    @GetMapping("/{key}")
    public byte[] getMedia(@PathVariable String key) {
        return mediaService.getMedia(key);
    }

    @DeleteMapping("/{key}")
    public void deleteMedia(@PathVariable String key) {
        mediaService.deleteMedia(key);
    }
}
