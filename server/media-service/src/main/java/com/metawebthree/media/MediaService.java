package com.metawebthree.media;

import com.baomidou.mybatisplus.core.toolkit.IdWorker;
import com.metawebthree.common.cloud.DefaultS3Config;
import com.metawebthree.common.cloud.DefaultS3Service;

import lombok.RequiredArgsConstructor;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;

import java.io.IOException;

@Service
@RequiredArgsConstructor
public class MediaService {
    private final DefaultS3Service s3Service;

    private final DefaultS3Config s3Config;

    public PutObjectResponse createMedia(String key, byte[] content) {
        return s3Service.putObject(s3Config.getName(), key, content);
    }

    public PutObjectResponse updateMedia(String key, byte[] content) {
        return s3Service.putObject(s3Config.getName(), key, content);
    }

    public byte[] getMedia(String key) {
        return s3Service.getObject(s3Config.getName(), key);
    }

    public void deleteMedia(String key) {
        s3Service.deleteObject(s3Config.getName(), key);
    }

    public String uploadFile(MultipartFile file) {
        String key = String.valueOf(IdWorker.getId());
        try {
            s3Service.putObject(s3Config.getName(), key, file.getBytes());
            return key;
        } catch (IOException e) {
            throw new RuntimeException("Failed to upload file", e);
        }
    }
}
