import json
from typing import List, Optional, Tuple, Union

from attr import dataclass
from pydantic import BaseModel, Field, model_validator
from pydantic import field_validator

class Target(BaseModel):
    content: str
    x: int
    y: int


class StimulusTrial(BaseModel):
    targets: List[Target] = Field(default=None, alias='stimulus')


class ViewHistory(BaseModel):
    page_index: int
    viewing_time: float


class ResponseDetail(BaseModel):
    dispositivo: Optional[str] = None
    mano: Optional[str] = None
    dispositivo_config: Optional[str] = None
    alcohol_drogas: Optional[str] = None
    tratamiento: Optional[str] = None
    usoDelPad: Optional[str] = None
    comentarioFinal: Optional[str] = None


class SubjectData(BaseModel):
    training_stimuli: Optional[List[StimulusTrial]] = Field(default=None, alias='train_stimuli')
    testing_stimuli: Optional[List[StimulusTrial]] = Field(default=None, alias='test_stimuli')
    position_coordinates: Optional[List[Tuple[float, float]]] = Field(default=None, alias='position')
    cursor_time: Optional[List[int]] = Field(default=None, alias='cursor_time')
    canvas_size: Optional[int] = None
    radius: Optional[int] = None
    x_y_clicked_position: Optional[List[int]] = Field(default=None, alias='x_y_clicked_position')
    response: Optional[Union[int, ResponseDetail, str]] = None
    accuracy: Optional[int] = None
    rt: Optional[float] = None

    #TODO GIAN: Charlar con gus un poco mas sobre que son estos 3 campos
    X_click: Optional[int] = None
    Y_click: Optional[int] = None
    T_click: Optional[int] = None

    # TODO GIAN: irrelevant fields
    success: Optional[bool] = None
    timeout: Optional[bool] = None
    failed_images: Optional[List[str]] = None
    failed_audio: Optional[List[str]] = None
    failed_video: Optional[List[str]] = None
    trial_type: Optional[str] = None
    trial_index: Optional[int] = None
    time_elapsed: Optional[int] = None
    internal_node_id: Optional[str] = None
    mean_rt: Optional[int] = None
    item_width_mm: Optional[float] = None
    item_height_mm: Optional[float] = None
    item_width_px: Optional[float] = None
    px2mm: Optional[float] = None
    view_dist_mm: Optional[float] = None
    item_width_deg: Optional[float] = None
    px2deg: Optional[float] = None
    scale_factor: Optional[float] = None
    win_width_deg: Optional[float] = None
    win_height_deg: Optional[float] = None
    screenX: Optional[int] = None
    screenY: Optional[int] = None
    innerX: Optional[int] = None
    innerY: Optional[int] = None
    view_history: Optional[List[ViewHistory]] = Field(default=None, alias='view_history')
    stimulus: Optional[str] = None
    FIX_COLOR: Optional[str] = None
    remaining_items_training: Optional[int] = None
    remaining_items_test: Optional[int] = None
    data: Optional[List[Target]] = Field(default=None)
    experimental_condition: Optional[str] = None
    screen_id: Optional[str] = None

    @field_validator('position_coordinates', mode='before')
    def parse_position_coordinates(cls, v):
        if isinstance(v, list):
            # Convert each string in the list to a tuple
            return [tuple(map(float, pos.strip('()').split(','))) for pos in v]
        return v


class Record(BaseModel):
    data: List[SubjectData]
    datetime: str

    @model_validator(mode='before')
    def parse_data(cls, values):
        data = values.get('data')
        if isinstance(data, str):
            values['data'] = json.loads(data)  # Convert string to list of SubjectData
        return values


class Experiment(BaseModel):
    subject_id: str = Field(alias='subject')
    experiment_status: str = Field(alias='status')
    records: List[Record]


class ExperimentRunCollection(BaseModel):
    experiments: List[Experiment] = Field(alias='experiment_runs')
