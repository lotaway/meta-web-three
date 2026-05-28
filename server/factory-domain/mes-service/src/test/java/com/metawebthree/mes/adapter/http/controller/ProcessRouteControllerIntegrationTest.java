package com.metawebthree.mes.adapter.http.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.repository.ProcessRouteRepository;
import com.metawebthree.mes.interfaces.dto.ProcessRouteDTO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import com.metawebthree.mes.config.TestConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.context.annotation.Import;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@Import(TestConfig.class)
@Transactional
@DisplayName("ProcessRoute REST API 集成测试")
class ProcessRouteControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private ProcessRouteRepository processRouteRepository;

    private ProcessRoute testRoute;

    @BeforeEach
    void setUp() {
        // 使用 H2 内存数据库，测试间自动清理
        testRoute = new ProcessRoute();
        testRoute.create("TEST-001", "测试工艺路线", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        ProcessRoute.ProcessStep step1 = new ProcessRoute.ProcessStep();
        step1.setStepNo(1);
        step1.setProcessCode("PC-001");
        step1.setProcessName("组装");
        step1.setWorkstationId("WS-001");
        step1.setStandardTime(300);
        steps.add(step1);
        
        ProcessRoute.ProcessStep step2 = new ProcessRoute.ProcessStep();
        step2.setStepNo(2);
        step2.setProcessCode("PC-002");
        step2.setProcessName("测试");
        step2.setWorkstationId("WS-002");
        step2.setStandardTime(120);
        step2.setPredecessorStepNo(1);
        steps.add(step2);
        
        testRoute.setSteps(steps);
        testRoute = processRouteRepository.save(testRoute);
    }

    @Nested
    @DisplayName("CRUD 操作测试")
    class CrudTests {
        
        @Test
        @DisplayName("创建工艺路线 - 成功")
        void testCreate_Success() throws Exception {
            ProcessRouteDTO.CreateRequest request = new ProcessRouteDTO.CreateRequest();
            request.setRouteCode("NEW-001");
            request.setRouteName("新工艺路线");
            request.setProductCode("P002");
            
            List<ProcessRouteDTO.ProcessStepDTO> steps = new ArrayList<>();
            ProcessRouteDTO.ProcessStepDTO stepDto = new ProcessRouteDTO.ProcessStepDTO();
            stepDto.setStepNo(1);
            stepDto.setProcessCode("PC-001");
            stepDto.setProcessName("组装");
            stepDto.setWorkstationId("WS-001");
            stepDto.setStandardTime(300);
            steps.add(stepDto);
            request.setSteps(steps);
            
            mockMvc.perform(post("/api/mes/process-route")
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(request)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.routeCode").value("NEW-001"))
                    .andExpect(jsonPath("$.routeName").value("新工艺路线"))
                    .andExpect(jsonPath("$.productCode").value("P002"))
                    .andExpect(jsonPath("$.status").value("DRAFT"));
        }
        
        @Test
        @DisplayName("创建工艺路线 - 验证失败（工序序号不连续）")
        void testCreate_ValidationFailed() throws Exception {
            ProcessRouteDTO.CreateRequest request = new ProcessRouteDTO.CreateRequest();
            request.setRouteCode("NEW-002");
            request.setRouteName("验证失败路线");
            request.setProductCode("P003");
            
            List<ProcessRouteDTO.ProcessStepDTO> steps = new ArrayList<>();
            ProcessRouteDTO.ProcessStepDTO stepDto = new ProcessRouteDTO.ProcessStepDTO();
            stepDto.setStepNo(1);
            stepDto.setProcessCode("PC-001");
            stepDto.setProcessName("组装");
            steps.add(stepDto);
            
            ProcessRouteDTO.ProcessStepDTO stepDto2 = new ProcessRouteDTO.ProcessStepDTO();
            stepDto2.setStepNo(3);  // 跳过了2
            stepDto2.setProcessCode("PC-002");
            stepDto2.setProcessName("测试");
            steps.add(stepDto2);
            request.setSteps(steps);
            
            mockMvc.perform(post("/api/mes/process-route")
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(request)))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.validationResult").value(false))
                    .andExpect(jsonPath("$.validationMessage").exists());
        }
        
        @Test
        @DisplayName("根据ID获取工艺路线 - 成功")
        void testGetById_Success() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId()))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.id").value(testRoute.getId().intValue()))
                    .andExpect(jsonPath("$.routeCode").value("TEST-001"))
                    .andExpect(jsonPath("$.routeName").value("测试工艺路线"));
        }
        
        @Test
        @DisplayName("根据ID获取工艺路线 - 不存在")
        void testGetById_NotFound() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/99999"))
                    .andExpect(status().isNotFound());
        }
        
        @Test
        @DisplayName("根据路线编码获取工艺路线")
        void testGetByCode() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/code/TEST-001"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.routeCode").value("TEST-001"));
        }
        
        @Test
        @DisplayName("更新工艺路线 - 成功")
        void testUpdate_Success() throws Exception {
            ProcessRouteDTO.UpdateRequest request = new ProcessRouteDTO.UpdateRequest();
            request.setRouteName("更新后的名称");
            request.setProductCode("P999");
            
            mockMvc.perform(put("/api/mes/process-route/" + testRoute.getId())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(request)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.routeName").value("更新后的名称"))
                    .andExpect(jsonPath("$.productCode").value("P999"));
        }
        
        @Test
        @DisplayName("删除工艺路线")
        void testDelete() throws Exception {
            mockMvc.perform(delete("/api/mes/process-route/" + testRoute.getId()))
                    .andExpect(status().isOk());
            
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId()))
                    .andExpect(status().isNotFound());
        }
        
        @Test
        @DisplayName("列表查询 - 无条件")
        void testList_All() throws Exception {
            mockMvc.perform(get("/api/mes/process-route"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$", hasSize(1)))
                    .andExpect(jsonPath("$[0].routeCode").value("TEST-001"));
        }
        
        @Test
        @DisplayName("列表查询 - 按状态筛选")
        void testList_ByStatus() throws Exception {
            // 测试查询 DRAFT 状态
            mockMvc.perform(get("/api/mes/process-route").param("status", "DRAFT"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$", hasSize(1)));
            
            // 测试查询 ACTIVE 状态（无数据）
            mockMvc.perform(get("/api/mes/process-route").param("status", "ACTIVE"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$", hasSize(0)));
        }
        
        @Test
        @DisplayName("根据产品编码查询")
        void testGetByProduct() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/product/P001"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$", hasSize(1)))
                    .andExpect(jsonPath("$[0].productCode").value("P001"));
        }
    }

    @Nested
    @DisplayName("状态转换测试")
    class StatusTransitionTests {
        
        @Test
        @DisplayName("激活工艺路线 - 成功")
        void testActivate_Success() throws Exception {
            mockMvc.perform(post("/api/mes/process-route/" + testRoute.getId() + "/activate"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.status").value("ACTIVE"));
        }
        
        @Test
        @DisplayName("激活工艺路线 - 验证失败")
        void testActivate_ValidationFailed() throws Exception {
            // 创建一个没有工序的工艺路线
            ProcessRoute invalidRoute = new ProcessRoute();
            invalidRoute.create("INVALID-001", "无效路线", "P001");
            // 不设置工序
            processRouteRepository.save(invalidRoute);
            
            mockMvc.perform(post("/api/mes/process-route/" + invalidRoute.getId() + "/activate"))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.validationResult").value(false));
        }
        
        @Test
        @DisplayName("归档工艺路线")
        void testArchive() throws Exception {
            // 先激活
            testRoute.activate();
            processRouteRepository.update(testRoute);
            // 重新查询获取最新状态
            testRoute = processRouteRepository.findById(testRoute.getId()).orElse(testRoute);
            
            mockMvc.perform(post("/api/mes/process-route/" + testRoute.getId() + "/archive"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.status").value("ARCHIVED"));
        }
        
        @Test
        @DisplayName("验证工艺路线")
        void testValidate() throws Exception {
            mockMvc.perform(post("/api/mes/process-route/" + testRoute.getId() + "/validate"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.validationResult").value(true))
                    .andExpect(jsonPath("$.validationMessage").value("验证通过"));
        }
    }

    @Nested
    @DisplayName("工序查询测试")
    class StepQueryTests {
        
        @Test
        @DisplayName("获取首道工序")
        void testGetFirstStep() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId() + "/first-step"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.stepNo").value(1))
                    .andExpect(jsonPath("$.processName").value("组装"));
        }
        
        @Test
        @DisplayName("获取下一道工序")
        void testGetNextStep() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId() + "/next-step/1"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.stepNo").value(2))
                    .andExpect(jsonPath("$.processName").value("测试"));
        }
        
        @Test
        @DisplayName("获取最后一道工序的下一道 - 返回404")
        void testGetNextStep_LastStep() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId() + "/next-step/2"))
                    .andExpect(status().isNotFound());
        }
    }
}