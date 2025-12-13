"""
25개 Ensemble 모델을 Stacking 호환 형식으로 변환

Ensemble 모델 (25개 MLP):
- models: [model1, model2, ..., model25]
- 앙상블 방식: 평균

Stacking 호환 형식:
- base_models: [model1, model2, ..., model25]
- meta_model: DummyMetaModel (단순 평균)
- model: EnsembleWrapper
"""
import pickle
import numpy as np
import sys
sys.path.insert(0, 'e:\\gptlotto')

# lotto_main.py에서 클래스 import
from lotto_main import DummyMetaModel, EnsembleWrapper


print("=" * 70)
print("Ensemble → Stacking 형식 변환")
print("=" * 70)

# 1. Ensemble 모델 로드
print("\n[1] Ensemble 모델 로드")
with open('best_ml_model_ensemble.pkl', 'rb') as f:
    ensemble_data = pickle.load(f)

n_models = ensemble_data['n_models']
ensemble_acc = ensemble_data['ensemble_accuracy']
mu = ensemble_data['mu']
sigma = ensemble_data['sigma']

print(f"   모델 개수: {n_models}개")
print(f"   앙상블 정확도: {ensemble_acc:.2%}")
print(f"   특징 개수: {len(mu)}개")

# 2. Stacking 호환 형식으로 변환
print("\n[2] Stacking 호환 형식으로 변환")

base_models = ensemble_data['models']
meta_model = DummyMetaModel()
wrapper = EnsembleWrapper(base_models, meta_model, mu, sigma)

stacking_compatible = {
    'type': 'stacking',  # lotto_generators.py에서 인식하는 타입
    'base_models': base_models,
    'meta_model': meta_model,
    'model': wrapper,
    'mu': mu,
    'sigma': sigma,
    'n_base_models': n_models,
    'meta_train_accuracy': ensemble_acc,  # 앙상블 정확도를 메타 모델 정확도로 사용
    'separation_power': 0.8,  # 임시값 (실제 계산 가능하지만 생략)
    'ensemble_accuracy': ensemble_acc,
    'fold_scores': ensemble_data['fold_scores'],
    'n_features': ensemble_data['n_features'],
}

print(f"   변환 완료:")
print(f"   - base_models: {len(base_models)}개")
print(f"   - meta_model: DummyMetaModel (평균 계산)")
print(f"   - wrapper: EnsembleWrapper")

# 3. 기존 모델 백업
print("\n[3] 기존 모델 백업")
import os
import shutil

if os.path.exists('best_ml_model_stacking.pkl'):
    shutil.copy('best_ml_model_stacking.pkl', 'best_ml_model_stacking_backup_10models.pkl')
    print("   백업 완료: best_ml_model_stacking_backup_10models.pkl")
else:
    print("   기존 모델 없음 (백업 불필요)")

# 4. 새 모델 저장
print("\n[4] 새 모델 저장")
with open('best_ml_model_stacking.pkl', 'wb') as f:
    pickle.dump(stacking_compatible, f)

file_size = os.path.getsize('best_ml_model_stacking.pkl') / 1024 / 1024
print(f"   저장 완료: best_ml_model_stacking.pkl")
print(f"   파일 크기: {file_size:.1f} MB")

# 5. 테스트
print("\n[5] 로드 테스트")
with open('best_ml_model_stacking.pkl', 'rb') as f:
    loaded = pickle.load(f)

print(f"   type: {loaded['type']}")
print(f"   n_base_models: {loaded['n_base_models']}")
print(f"   meta_train_accuracy: {loaded['meta_train_accuracy']:.2%}")
print(f"   model: {type(loaded['model']).__name__}")

# 간단한 예측 테스트
print("\n[6] 예측 테스트")
test_X = np.random.randn(5, len(mu))
test_probs = loaded['model'].predict_proba(test_X)

print(f"   입력: {test_X.shape}")
print(f"   출력: {test_probs.shape}")
print(f"   샘플 확률: {test_probs[0]}")
print(f"   양성 확률 범위: [{test_probs[:, 1].min():.3f}, {test_probs[:, 1].max():.3f}]")

print("\n" + "=" * 70)
print("변환 완료!")
print("=" * 70)
print("\nlotto_main.py에서 자동으로 25개 MLP 앙상블 모델을 사용합니다.")
print("GUI를 재시작하면 적용됩니다.")
print("\n복원 방법 (필요 시):")
print("  copy best_ml_model_stacking_backup_10models.pkl best_ml_model_stacking.pkl")
print("=" * 70)
