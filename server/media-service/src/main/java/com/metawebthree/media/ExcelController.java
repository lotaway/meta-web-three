package com.metawebthree.media;

import org.springframework.http.HttpHeaders;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import com.metawebthree.common.dto.ApiResponse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;

import java.io.UnsupportedEncodingException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;

@RestController
@RequestMapping("/excel")
@Tag(name = "Excel Management")
public class ExcelController {

    private final ExcelService excelService;

    public ExcelController(ExcelService excelService) {
        this.excelService = excelService;
    }

    @GetMapping("/template")
    @Operation(summary = "Show/Download Excel import/upload template")
    public ResponseEntity<String> downloadTemplate() throws UnsupportedEncodingException {
        String fileUrl = excelService.generateTemplate();
        return ResponseEntity.ok()
                // .contentType(MediaType.parseMediaType("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"))
                // .header(HttpHeaders.CONTENT_DISPOSITION, "attachment; filename=" + fileName)
                .contentType(MediaType.TEXT_PLAIN)
                .body(fileUrl);
    }

    @PostMapping("/import/url")
    public ApiResponse<?> importExcel(@RequestParam String excelUrl) {
        excelService.processExcelData(excelUrl);
        return ApiResponse.success();
    }

    @PostMapping("/import/file")
    @Operation(summary = "Upload and import Excel file")
    public ApiResponse<?> importExcelFile(@RequestParam("file") MultipartFile file) {
        excelService.processExcelFile(file);
        return ApiResponse.success();
    }
}
