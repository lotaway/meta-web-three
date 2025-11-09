package com.metawebthree.media;

import org.junit.Assert;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.junit4.SpringRunner;
import org.springframework.web.client.RestTemplate;

import com.github.yulichang.wrapper.MPJLambdaWrapper;
import com.metawebthree.media.DO.ArtWorkDO;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ComponentScan(basePackages = {
        "com.metawebthree.media",
        "com.metawebthree.common"
})
@EnableAutoConfiguration
public class ExcelServiceTest {

    @Autowired
    private ExcelService excelService;

    @Autowired
    private ArtWorkMapper artWorkMapper;

    @Test
    public void testGenerateExcelTemplate() {
        String fileUrl = excelService.generateTemplate();
        Assert.assertNotNull(fileUrl);
        Assert.assertTrue(fileUrl.startsWith("http"));
        var restTemplate = new RestTemplate();
        ResponseEntity<byte[]> response = restTemplate.getForEntity(fileUrl, byte[].class);
        Assert.assertEquals(HttpStatus.OK, response.getStatusCode());
        HttpHeaders headers = response.getHeaders();
        String contentType = headers.getContentType().toString();
        Assert.assertTrue("Content-Type should indicate Excel file",
                contentType.contains("spreadsheetml") || contentType.contains("excel"));
        byte[] content = response.getBody();
        Assert.assertTrue("File content too short", content.length >= 2);
    }

    @Test
    public void testImportExcel() {
        var wrapper = new MPJLambdaWrapper<ArtWorkDO>().select(ArtWorkDO::getId);
        Long originCount = artWorkMapper.selectCount(wrapper);
        excelService.processExcelData("your-excel-file-download-url");
        Long count = artWorkMapper
                .selectCount(wrapper);
        Assert.assertTrue(count > originCount);
    }
}