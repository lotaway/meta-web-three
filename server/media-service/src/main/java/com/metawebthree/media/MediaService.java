package com.metawebthree.media;

import com.metawebthree.common.cloud.DefaultS3Buckets;
import com.metawebthree.common.cloud.DefaultS3Service;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import software.amazon.awssdk.services.s3.model.PutObjectResponse;

import java.io.IOException;

@Service
public class MediaService {
    private final DefaultS3Service s3Service;
    private final DefaultS3Buckets s3Bucket;

    public MediaService(DefaultS3Service s3Service, DefaultS3Buckets s3Bucket) {
        this.s3Service = s3Service;
        this.s3Bucket = s3Bucket;
    }

    public PutObjectResponse createMedia(String key, byte[] content) {
        return s3Service.putObject(s3Bucket.getName(), key, content);
    }

    public PutObjectResponse updateMedia(String key, byte[] content) {
        return s3Service.putObject(s3Bucket.getName(), key, content);
    }

    public byte[] getMedia(String key) {
        return s3Service.getObject(s3Bucket.getName(), key);
    }

    public void deleteMedia(String key) {
        s3Service.deleteObject(s3Bucket.getName(), key);
    }

    public String uploadFile(MultipartFile file) {
        String key = IdWorker.getId().toString();
        try {
            s3Service.putObject(s3Bucket.getName(), key, file.getBytes());
            return key;
        } catch (IOException e) {
            throw new RuntimeException("Failed to upload file", e);
        }
    }
}
