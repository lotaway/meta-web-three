package com.metawebthree.server.dao;

import org.springframework.stereotype.Repository;

import java.util.Objects;

@Repository
public class Resource {
    public String getFile(String filePath) {
        return Objects.requireNonNull(this.getClass().getClassLoader().getResource(filePath)).getFile();
    }
}
