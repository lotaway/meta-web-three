package com.metawebthree.media.setting;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class SettingService {

    @Autowired
    private ResourceRepository resourceRepository;

    public String getDefaultConfig() {
        return resourceRepository.getFile("static/settings.xml");
    }
}
