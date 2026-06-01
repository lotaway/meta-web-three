package com.metawebthree.review.interfaces.controller;

import com.metawebthree.review.application.dto.ReviewCreateDTO;
import com.metawebthree.review.application.dto.ReviewDTO;
import com.metawebthree.review.application.dto.ReviewReplyDTO;
import com.metawebthree.review.application.service.ReviewApplicationService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/review")
public class ReviewController {

    private final ReviewApplicationService reviewService;

    public ReviewController(ReviewApplicationService reviewService) {
        this.reviewService = reviewService;
    }

    /**
     * Create a review
     */
    @PostMapping("/create")
    public ResponseEntity<ReviewDTO> create(
            @RequestBody ReviewCreateDTO createDTO,
            @RequestHeader("X-User-Id") Long userId) {
        ReviewDTO result = reviewService.create(createDTO, userId);
        return ResponseEntity.ok(result);
    }

    /**
     * Approve review (admin)
     */
    @PostMapping("/{id}/approve")
    public ResponseEntity<ReviewDTO> approve(@PathVariable Long id) {
        ReviewDTO result = reviewService.approve(id, true);
        return ResponseEntity.ok(result);
    }

    /**
     * Reject review (admin)
     */
    @PostMapping("/{id}/reject")
    public ResponseEntity<ReviewDTO> reject(@PathVariable Long id) {
        ReviewDTO result = reviewService.approve(id, false);
        return ResponseEntity.ok(result);
    }

    /**
     * Reply to review (store)
     */
    @PostMapping("/reply")
    public ResponseEntity<ReviewDTO> reply(@RequestBody ReviewReplyDTO replyDTO) {
        ReviewDTO result = reviewService.reply(replyDTO);
        return ResponseEntity.ok(result);
    }

    /**
     * Like a review
     */
    @PostMapping("/{id}/like")
    public ResponseEntity<ReviewDTO> like(@PathVariable Long id) {
        ReviewDTO result = reviewService.like(id);
        return ResponseEntity.ok(result);
    }

    /**
     * Get review by ID
     */
    @GetMapping("/{id}")
    public ResponseEntity<ReviewDTO> getById(@PathVariable Long id) {
        ReviewDTO result = reviewService.getById(id);
        return ResponseEntity.ok(result);
    }

    /**
     * Get reviews by product ID
     */
    @GetMapping("/product/{productId}")
    public ResponseEntity<List<ReviewDTO>> getByProductId(
            @PathVariable Long productId,
            @RequestParam(required = false) String sortBy) {
        List<ReviewDTO> result;
        if (sortBy != null) {
            result = reviewService.getByProductIdSorted(productId, sortBy);
        } else {
            result = reviewService.getByProductId(productId);
        }
        return ResponseEntity.ok(result);
    }

    /**
     * Get reviews by user ID
     */
    @GetMapping("/user/{userId}")
    public ResponseEntity<List<ReviewDTO>> getByUserId(@PathVariable Long userId) {
        List<ReviewDTO> result = reviewService.getByUserId(userId);
        return ResponseEntity.ok(result);
    }

    /**
     * Get reviews by store ID
     */
    @GetMapping("/store/{storeId}")
    public ResponseEntity<List<ReviewDTO>> getByStoreId(@PathVariable Long storeId) {
        List<ReviewDTO> result = reviewService.getByStoreId(storeId);
        return ResponseEntity.ok(result);
    }

    /**
     * Get all reviews (admin)
     */
    @GetMapping("/list")
    public ResponseEntity<List<ReviewDTO>> getAll() {
        List<ReviewDTO> result = reviewService.getAll();
        return ResponseEntity.ok(result);
    }
}