package com.metawebthree.cs.interfaces.web;

import com.metawebthree.common.dto.ApiResponse;
import com.metawebthree.cs.application.FaqService;
import com.metawebthree.cs.domain.model.Faq;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/cs/faq")
@Tag(name = "FAQ Controller", description = "知识库FAQ接口")
public class FaqController {
    private final FaqService faqService;

    public FaqController(FaqService faqService) {
        this.faqService = faqService;
    }

    @Operation(summary = "创建FAQ")
    @PostMapping("/create")
    public ApiResponse<Faq> create(@RequestBody FaqRequest request) {
        return ApiResponse.success(
                faqService.createFaq(request.getQuestion(), request.getAnswer(),
                        request.getCategory(), request.getKeywords()));
    }

    @Operation(summary = "更新FAQ")
    @PostMapping("/update")
    public ApiResponse<Faq> update(@RequestParam Long id, @RequestBody FaqRequest request) {
        return ApiResponse.success(
                faqService.updateFaq(id, request.getQuestion(), request.getAnswer(),
                        request.getCategory(), request.getKeywords(),
                        request.getEnabled(), request.getPriority()));
    }

    @Operation(summary = "删除FAQ")
    @DeleteMapping("/delete")
    public ApiResponse<Void> delete(@RequestParam Long id) {
        faqService.deleteFaq(id);
        return ApiResponse.success();
    }

    @Operation(summary = "根据ID查询FAQ")
    @GetMapping("/get")
    public ApiResponse<Faq> get(@RequestParam Long id) {
        return ApiResponse.success(faqService.getFaq(id).orElse(null));
    }

    @Operation(summary = "查询所有FAQ")
    @GetMapping("/list")
    public ApiResponse<List<Faq>> list() {
        return ApiResponse.success(faqService.getAllFaqs());
    }

    @Operation(summary = "根据分类查询")
    @GetMapping("/listByCategory")
    public ApiResponse<List<Faq>> listByCategory(@RequestParam String category) {
        return ApiResponse.success(faqService.getFaqsByCategory(category));
    }

    @Operation(summary = "关键字搜索")
    @GetMapping("/search")
    public ApiResponse<List<Faq>> search(@RequestParam String keyword) {
        return ApiResponse.success(faqService.searchByKeyword(keyword));
    }

    @Operation(summary = "智能匹配回复")
    @GetMapping("/match")
    public ApiResponse<Faq> match(@RequestParam String query) {
        return ApiResponse.success(faqService.searchAndMatch(query));
    }

    @Operation(summary = "获取热门FAQ")
    @GetMapping("/top")
    public ApiResponse<List<Faq>> top(@RequestParam(defaultValue = "10") int limit) {
        return ApiResponse.success(faqService.searchTopRelevance(limit));
    }

    @Operation(summary = "获取总数")
    @GetMapping("/count")
    public ApiResponse<Long> count() {
        return ApiResponse.success(faqService.getTotalCount());
    }

    public static class FaqRequest {
        private String question;
        private String answer;
        private String category;
        private List<String> keywords;
        private Boolean enabled;
        private Integer priority;

        public String getQuestion() { return question; }
        public void setQuestion(String question) { this.question = question; }
        public String getAnswer() { return answer; }
        public void setAnswer(String answer) { this.answer = answer; }
        public String getCategory() { return category; }
        public void setCategory(String category) { this.category = category; }
        public List<String> getKeywords() { return keywords; }
        public void setKeywords(List<String> keywords) { this.keywords = keywords; }
        public Boolean getEnabled() { return enabled; }
        public void setEnabled(Boolean enabled) { this.enabled = enabled; }
        public Integer getPriority() { return priority; }
        public void setPriority(Integer priority) { this.priority = priority; }
    }
}