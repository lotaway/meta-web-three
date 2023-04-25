package com.metawebthree.controller;

import com.metawebthree.service.impl.UserService;
import com.metawebthree.service.impl.UserTypeService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@Slf4j
@RestController
@RequestMapping("/userType")
class UserTypeController {

    private final UserTypeService userTypeService;

    UserTypeController(UserTypeService userTypeService) {
        this.userTypeService = userTypeService;
    }

    @GetMapping
    String getIndex() {
        return "Hello, user type controller.";
    }

    @GetMapping("/list")
    String getRegisterType() {
        log.info("into userType list controller");
        return userTypeService.getList().toString();
    }
}