package com.metawebthree;

import org.apache.dubbo.config.annotation.DubboReference;
import org.junit.Assert;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.test.context.junit4.SpringRunner;

import com.metawebthree.common.generated.rpc.RiskScorerService;
import com.metawebthree.common.generated.rpc.TestRequest;
import com.metawebthree.common.generated.rpc.TestResponse;

@RunWith(SpringRunner.class)
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.DEFINED_PORT)
@ComponentScan(basePackages = {
        "com.metawebthree",
        "com.metawebthree.common"
})
@EnableAutoConfiguration
class PaymentServiceApplicationTests {

    @DubboReference(protocol = "tri")
    private RiskScorerService riskScorerService;

    @Test
    void testRPCService() {
        TestResponse response = riskScorerService.test(TestRequest.newBuilder().build());
        System.out.println(String.format("RPC Service test result: %s", response.getResult()));
        Assert.assertTrue(response.getResult() > 0);
    }

}