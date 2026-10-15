package com.metawebthree.media.application.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;

@Data

@AllArgsConstructor
@Builder
public class DanmuMessageDTO {
        public String nickname;
        public String type;
        public String username;
        public String content;
        public String color;
        public String position;
        public int size;
}
