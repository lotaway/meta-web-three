package com.metawebthree.mes.adapter.http.controller;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.metawebthree.mes.domain.entity.ProcessRoute;
import com.metawebthree.mes.domain.repository.ProcessRouteRepository;
import com.metawebthree.mes.infrastructure.event.MesEventPublisher;
import com.metawebthree.mes.interfaces.dto.ProcessRouteDTO;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.context.ActiveProfiles;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.transaction.annotation.Transactional;

import java.util.ArrayList;
import java.util.List;

import static org.hamcrest.Matchers.*;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doNothing;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@Transactional
@DisplayName("ProcessRoute REST API Integration Test")
class ProcessRouteControllerIntegrationTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private ObjectMapper objectMapper;

    @Autowired
    private ProcessRouteRepository processRouteRepository;

    @MockBean
    private MesEventPublisher mesEventPublisher;

    private ProcessRoute testRoute;

    @BeforeEach
    void setUp() {
        doNothing().when(mesEventPublisher).publishWorkOrderCreated(anyLong(), anyString());
        doNothing().when(mesEventPublisher).publishWorkOrderReleased(anyLong());
        doNothing().when(mesEventPublisher).publishWorkOrderStarted(anyLong());
        doNothing().when(mesEventPublisher).publishWorkOrderCompleted(anyLong());


        testRoute = new ProcessRoute();
        testRoute.create("TEST-001", "Test Route", "P001");
        
        List<ProcessRoute.ProcessStep> steps = new ArrayList<>();
        ProcessRoute.ProcessStep step1 = new ProcessRoute.ProcessStep();
        step1.setStepNo(1);
        step1.setProcessCode("PC-001");
        step1.setProcessName("Assembly");
        step1.setWorkstationId(1L);
        step1.setStandardTime(300);
        steps.add(step1);
        
        ProcessRoute.ProcessStep step2 = new ProcessRoute.ProcessStep();
        step2.setStepNo(2);
        step2.setProcessCode("PC-002");
        step2.setProcessName("Testing");
        step2.setWorkstationId(2L);
        step2.setStandardTime(120);
        step2.setPredecessorStepNo(1);
        steps.add(step2);
        
        testRoute.setSteps(steps);
        testRoute = processRouteRepository.save(testRoute);
    }

    @Nested
    @DisplayName("CRUD Operations")
    class CrudTests {
        
        @Test
        @DisplayName("Create ProcessRoute - Success")
        void testCreate_Success() throws Exception {
            ProcessRouteDTO.CreateRequest request = new ProcessRouteDTO.CreateRequest();
            request.setRouteCode("NEW-001");
            request.setRouteName("New Route");
            request.setProductCode("P002");
            
            List<ProcessRouteDTO.ProcessStepDTO> steps = new ArrayList<>();
            ProcessRouteDTO.ProcessStepDTO stepDto = new ProcessRouteDTO.ProcessStepDTO();
            stepDto.setStepNo(1);
            stepDto.setProcessCode("PC-001");
            stepDto.setProcessName("Assembly");
            stepDto.setWorkstationId(1L);
            stepDto.setStandardTime(300);
            steps.add(stepDto);
            request.setSteps(steps);
            
            mockMvc.perform(post("/api/mes/process-route")
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(request)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.routeCode").value("NEW-001"))
                    .andExpect(jsonPath("$.routeName").value("New Route"))
                    .andExpect(jsonPath("$.productCode").value("P002"))
                    .andExpect(jsonPath("$.status").value("DRAFT"));
        }
        
        @Test
        @DisplayName("Create ProcessRoute - Validation Failed (Non-consecutive Step No)")
        void testCreate_ValidationFailed() throws Exception {
            ProcessRouteDTO.CreateRequest request = new ProcessRouteDTO.CreateRequest();
            request.setRouteCode("NEW-002");
            request.setRouteName("Invalid Route");
            request.setProductCode("P003");
            
            List<ProcessRouteDTO.ProcessStepDTO> steps = new ArrayList<>();
            ProcessRouteDTO.ProcessStepDTO stepDto = new ProcessRouteDTO.ProcessStepDTO();
            stepDto.setStepNo(1);
            stepDto.setProcessCode("PC-001");
            stepDto.setProcessName("Assembly");
            steps.add(stepDto);
            
            ProcessRouteDTO.ProcessStepDTO stepDto2 = new ProcessRouteDTO.ProcessStepDTO();
            stepDto2.setStepNo(3);
            stepDto2.setProcessCode("PC-002");
            stepDto2.setProcessName("Testing");
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
        @DisplayName("Get ProcessRoute By ID - Success")
        void testGetById_Success() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId()))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.id").value(testRoute.getId().intValue()))
                    .andExpect(jsonPath("$.routeCode").value("TEST-001"))
                    .andExpect(jsonPath("$.routeName").value("Test Route"));
        }
        
        @Test
        @DisplayName("Get ProcessRoute By ID - Not Found")
        void testGetById_NotFound() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/99999"))
                    .andExpect(status().isNotFound());
        }
        
        @Test
        @DisplayName("Get ProcessRoute By Code")
        void testGetByCode() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/code/TEST-001"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.routeCode").value("TEST-001"));
        }
        
        @Test
        @DisplayName("Update ProcessRoute - Success")
        void testUpdate_Success() throws Exception {
            ProcessRouteDTO.UpdateRequest request = new ProcessRouteDTO.UpdateRequest();
            request.setRouteName("Updated Name");
            request.setProductCode("P999");
            
            mockMvc.perform(put("/api/mes/process-route/" + testRoute.getId())
                    .contentType(MediaType.APPLICATION_JSON)
                    .content(objectMapper.writeValueAsString(request)))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.routeName").value("Updated Name"))
                    .andExpect(jsonPath("$.productCode").value("P999"));
        }
        
        @Test
        @DisplayName("Delete ProcessRoute")
        void testDelete() throws Exception {
            mockMvc.perform(delete("/api/mes/process-route/" + testRoute.getId()))
                    .andExpect(status().isOk());
            
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId()))
                    .andExpect(status().isNotFound());
        }
        
        @Test
        @DisplayName("List All ProcessRoutes")
        void testList_All() throws Exception {
            mockMvc.perform(get("/api/mes/process-route"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$", hasSize(1)))
                    .andExpect(jsonPath("$[0].routeCode").value("TEST-001"));
        }
        
        @Test
        @DisplayName("List ProcessRoutes By Status")
        void testList_ByStatus() throws Exception {
            mockMvc.perform(get("/api/mes/process-route").param("status", "DRAFT"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$", hasSize(1)));
            
            mockMvc.perform(get("/api/mes/process-route").param("status", "ACTIVE"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$", hasSize(0)));
        }
        
        @Test
        @DisplayName("Get ProcessRoutes By Product Code")
        void testGetByProduct() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/product/P001"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$", hasSize(1)))
                    .andExpect(jsonPath("$[0].productCode").value("P001"));
        }
    }

    @Nested
    @DisplayName("Status Transition Tests")
    class StatusTransitionTests {
        
        @Test
        @DisplayName("Activate ProcessRoute - Success")
        void testActivate_Success() throws Exception {
            mockMvc.perform(post("/api/mes/process-route/" + testRoute.getId() + "/activate"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.status").value("ACTIVE"));
        }
        
        @Test
        @DisplayName("Activate ProcessRoute - Validation Failed")
        void testActivate_ValidationFailed() throws Exception {
            ProcessRoute invalidRoute = new ProcessRoute();
            invalidRoute.create("INVALID-001", "Invalid Route", "P001");
            processRouteRepository.save(invalidRoute);
            
            mockMvc.perform(post("/api/mes/process-route/" + invalidRoute.getId() + "/activate"))
                    .andExpect(status().isBadRequest())
                    .andExpect(jsonPath("$.validationResult").value(false));
        }
        
        @Test
        @DisplayName("Archive ProcessRoute")
        void testArchive() throws Exception {
            testRoute.activate();
            processRouteRepository.update(testRoute);
            testRoute = processRouteRepository.findById(testRoute.getId()).orElse(testRoute);
            
            mockMvc.perform(post("/api/mes/process-route/" + testRoute.getId() + "/archive"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.status").value("ARCHIVED"));
        }
        
        @Test
        @DisplayName("Validate ProcessRoute")
        void testValidate() throws Exception {
            mockMvc.perform(post("/api/mes/process-route/" + testRoute.getId() + "/validate"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.validationResult").value(true))
                    .andExpect(jsonPath("$.validationMessage").value("Validation Passed"));
        }
    }

    @Nested
    @DisplayName("Step Query Tests")
    class StepQueryTests {
        
        @Test
        @DisplayName("Get First Step")
        void testGetFirstStep() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId() + "/first-step"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.stepNo").value(1))
                    .andExpect(jsonPath("$.processName").value("Assembly"));
        }
        
        @Test
        @DisplayName("Get Next Step")
        void testGetNextStep() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId() + "/next-step/1"))
                    .andExpect(status().isOk())
                    .andExpect(jsonPath("$.stepNo").value(2))
                    .andExpect(jsonPath("$.processName").value("Testing"));
        }
        
        @Test
        @DisplayName("Get Next Step From Last Step - Returns 404")
        void testGetNextStep_LastStep() throws Exception {
            mockMvc.perform(get("/api/mes/process-route/" + testRoute.getId() + "/next-step/2"))
                    .andExpect(status().isNotFound());
        }
    }
}