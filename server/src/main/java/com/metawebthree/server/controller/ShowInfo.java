package com.metawebthree.server.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.bind.annotation.RequestMapping;
import com.config.InitScanner;

@RestController
public class ShowInfo {
    @Value("${name}")
    private String name;

    @Autowired
    private Environment env;

    @RequestMapping("/name")
    public String name() {
        return name;
    }

    private final Author author;

    public ShowInfo(Author author) {
        this.author = author;
    }

    @RequestMapping("/author")
    public String getAuthor() {
        return this.author.toString();
    }

    @RequestMapping("/scanner")
    public String scanner() throws Exception {
        return InitScanner.getInfo();
    }

    @RequestMapping(path = "/showConfig", method = RequestMethod.GET)
    public String showConfig() {
        return "showConfig";
    }
}
