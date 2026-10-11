package com.metawebthree.recommendation.domain.aishopping.entity;

/**
 * AI shopping match result, shared by smart match and image search.
 */
public class AiProductMatch {

    private Long productId;
    private String name;
    private String pic;
    private String price;
    private Double score;
    private String reason;

    public AiProductMatch() {
    }

    public AiProductMatch(Long productId, String name, String pic, String price, Double score, String reason) {
        this.productId = productId;
        this.name = name;
        this.pic = pic;
        this.price = price;
        this.score = score;
        this.reason = reason;
    }

    public Long getProductId() {
        return productId;
    }

    public void setProductId(Long productId) {
        this.productId = productId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getPic() {
        return pic;
    }

    public void setPic(String pic) {
        this.pic = pic;
    }

    public String getPrice() {
        return price;
    }

    public void setPrice(String price) {
        this.price = price;
    }

    public Double getScore() {
        return score;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }
}
