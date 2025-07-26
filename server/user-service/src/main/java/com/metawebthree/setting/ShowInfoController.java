package com.metawebthree.setting;

import com.metawebthree.common.ApiResponse;
import com.metawebthree.common.ProjectAuthorVO;
import com.metawebthree.common.ShowErrorArgsVO;
import jakarta.servlet.http.HttpServletRequest;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.env.Environment;
import org.springframework.web.bind.annotation.*;
import com.config.InitScanner;

//import java.io.Serial;
import java.io.Serializable;

@Slf4j
@RequestMapping("/showInfo")
@RestController
public class ShowInfoController implements Serializable {

//    @Serial
    private static final long serialVersionUID = 1L;

    @Value("${name}")
    private String name;

    @Autowired
    private Environment env;

    @Autowired
    private SettingService settingService;

    private final ProjectAuthorVO projectAuthorVo;

    public ShowInfoController(ProjectAuthorVO projectAuthorVo) {
        this.projectAuthorVo = projectAuthorVo;
    }

    @RequestMapping("/name")
    public String name() {
        return name;
    }

    @RequestMapping("/author")
    public ProjectAuthorVO getAuthor() {
        return this.projectAuthorVo;
    }

    @RequestMapping("/scanner")
    public String scanner(HttpServletRequest request) throws Exception {
        String showType = request.getParameter("showType");
        return InitScanner.getInfo();
    }

    @RequestMapping(path = "/showConfig", method = RequestMethod.POST)
    public ApiResponse showConfig(@RequestParam String type) {
        return ApiResponse.success("show config");
    }

    @RequestMapping("/showError")
    public String showError(ShowErrorArgsVO showErrorArgsVo) throws Exception {
        boolean result = InitScanner.errorOutputToFile("Nothing error");
        System.out.println("error output to file result: " + result);
        return InitScanner.getErrorLog("error/error.log");
    }

    @RequestMapping("/config/{id}")
    public String testPath(@PathVariable int id) {
        String fileConfig = settingService.getTestConfig();
        return "test path parameter id is:" + id + ", and get config file test:" + fileConfig;
    }
}
