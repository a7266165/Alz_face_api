"""
API 請求資料模型
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class QuestionnaireData(BaseModel):
    """6QDS 問卷資料"""
    
    # 人口學資料
    age: int = Field(
        ...,
        ge=0,
        le=150,
        description="年齡（歲）"
    )
    
    gender: int = Field(
        ...,
        ge=0,
        le=1,
        description="性別 (0=女性, 1=男性)"
    )
    
    education_years: int = Field(
        ...,
        ge=0,
        le=30,
        description="教育年數"
    )
    
    # 問卷題目
    q1: int = Field(..., ge=0, le=1, description="題目 1 (0-1)")
    q2: int = Field(..., ge=0, le=2, description="題目 2 (0-2)")
    q3: int = Field(..., ge=0, le=2, description="題目 3 (0-2)")
    q4: int = Field(..., ge=0, le=1, description="題目 4 (0-1)")
    q5: int = Field(..., ge=0, le=1, description="題目 5 (0-1)")
    q6: int = Field(..., ge=0, le=1, description="題目 6 (0-1)")
    q7: int = Field(..., ge=0, le=1, description="題目 7 (0-1)")
    q8: int = Field(..., ge=0, le=1, description="題目 8 (0-1)")
    q9: int = Field(..., ge=0, le=1, description="題目 9 (0-1)")
    q10: int = Field(..., ge=0, le=1, description="題目 10 (0-1)")
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v: int) -> int:
        """驗證年齡合理性"""
        if v < 18:
            import logging
            logging.warning(f"年齡偏低: {v} 歲")
        return v
    
    def to_feature_array(self) -> list:
        """轉換為模型輸入格式（給 6QDS 模型用）"""
        return [
            self.age,
            self.gender,
            self.education_years,
            self.q1, self.q2, self.q3, self.q4, self.q5,
            self.q6, self.q7, self.q8, self.q9, self.q10
        ]
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 70,
                "gender": 1,
                "education_years": 12,
                "q1": 1,
                "q2": 0,
                "q3": 2,
                "q4": 1,
                "q5": 0,
                "q6": 1,
                "q7": 0,
                "q8": 1,
                "q9": 0,
                "q10": 1
            }
        }