"""Validated runtime limits; dashboard overrides live in the plugin's own store."""

from maibot_sdk import Field, PluginConfigBase
from pydantic import model_validator


class Settings(PluginConfigBase):
    daily_enabled: bool = Field(default=True, description="每天启动后自动漫游；未导入审美基础时不运行")
    daily_hour: int = Field(default=0, ge=0, le=23, description="每日最早运行小时，Asia/Singapore")
    daily_authors: int = Field(default=3, ge=1, le=10, description="每天希望深入的不同摄影师数量，允许少选")
    preview_min: int = Field(default=30, ge=1, description="每天至少希望真实预览的不同图片数；无法达到时说明原因")
    preview_max: int = Field(default=60, ge=1, description="每天真实预览的图片上限")
    detail_min: int = Field(default=10, ge=1, description="从预览中挑选深入观看的目标下限，不要求收藏")
    detail_max: int = Field(default=20, ge=1, description="深入观看的图片上限")
    max_images: int = Field(default=700, ge=10, le=5000, description="自动收藏图片硬上限")
    max_mb: int = Field(default=1024, ge=16, le=10240, description="自动收藏磁盘预算 MiB")
    idle_days: int = Field(default=4, ge=1, le=365, description="双方均低关注图片的闲置清理天数")
    vision_task: str = Field(default="vlm", min_length=1, max_length=80, description="Host 视觉模型任务名称")
    text_task: str = Field(default="utils", min_length=1, max_length=80, description="Host 决策与整理模型任务名称")
    image_embedding_task: str = Field(default="image_embedding", min_length=1, max_length=80,
                                      description="Host 图片嵌入任务，用于关联历史作品")
    share_enabled: bool = Field(default=False, description="启用向选定群分享；需先选群")
    target_stream_id: str = Field(default="", max_length=128, description="用户选择的真实群聊流")
    share_count: int = Field(default=2, ge=1, description="每次分享的图片数量上限，由用户设置；不限制每天总量")
    exploration_ratio: float = Field(default=0.2, ge=0.05, le=0.5, description="给陌生来源和新方向保留的探索比例")

    @model_validator(mode="after")
    def validate_viewing_ranges(self) -> "Settings":
        if self.preview_min > self.preview_max or self.detail_min > self.detail_max:
            raise ValueError("观看数量下限不能超过上限")
        if self.detail_max > self.preview_max:
            raise ValueError("精看图片从预览中选择，精看上限不能超过预览上限")
        return self


class PluginSection(PluginConfigBase):
    enabled: bool = Field(default=True, description="启用摄影漫游")
    config_version: str = Field(default="0.3.0", description="配置版本")


class PhotographyConfig(PluginConfigBase):
    plugin: PluginSection = Field(default_factory=PluginSection)
    photography: Settings = Field(default_factory=Settings)
