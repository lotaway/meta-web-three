package com.metawebthree.service.impl;

import com.metawebthree.dao.Resource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class SettingService {
    @Autowired
    private Resource resource;

    public String getTestConfig() {
        return resource.getFile("static/test.xml");
    }
}
