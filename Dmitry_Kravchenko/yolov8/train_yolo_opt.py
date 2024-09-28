import optuna
import optuna.visualization as vis
from ultralytics import YOLO
from datetime import datetime

def objective(trial):
    # Определяем пространство поиска гиперпараметров
    epochs = trial.suggest_int('epochs', 50, 300)
    lr0 = trial.suggest_float('lr0', 1e-5, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 0.6, 0.98)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    optimizer = trial.suggest_categorical('optimizer', ['SGD', 'Adam', 'AdamW'])

    # Загружаем модель
    model = YOLO('yolov8m.pt')

    # Запускаем обучение с выбранными гиперпараметрами
    model.train(
        data='C:/Dima/Projects/Cuda/LOM/dataset.yaml',
        epochs=epochs,
        imgsz=640,
        batch=-1,  # Автоматический выбор batch_size
        lr0=lr0,
        weight_decay=weight_decay,
        momentum=momentum,
        optimizer=optimizer,
        freeze=[0],
        warmup_epochs=2.0,
        cos_lr=True,
        rect=False,
        augment=True,
        patience=300,
        device='cuda'
    )

    # Оценка модели на валидационном наборе
    metrics = model.val()

    # Возвращаем mAP 0.5-0.95 (для оптимизации)
    return metrics.box.maps().mean()  # Среднее значение mAP по всем классам и IoU

if __name__ == "__main__":
    # Создаем объект для исследования (study)
    study = optuna.create_study(direction='maximize')
    
    # Запускаем оптимизацию
    study.optimize(objective, n_trials=50)

    # Выводим лучшие гиперпараметры
    print(f'Лучшие гиперпараметры: {study.best_params}')
    
    # Сохраняем результаты исследования
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study.trials_dataframe().to_csv(f'C:/Dima/Projects/Cuda/LOM/optuna_results_{timestamp}.csv', index=False)

    # Визуализация важности гиперпараметров
    param_importances = vis.plot_param_importances(study)
    param_importances.write_html(f'C:/Dima/Projects/Cuda/LOM/optuna_visualizations/param_importances_{timestamp}.html')

    # Визуализация истории оптимизации
    optimization_history = vis.plot_optimization_history(study)
    optimization_history.write_html(f'C:/Dima/Projects/Cuda/LOM/optuna_visualizations/optimization_history_{timestamp}.html')

    # Визуализация зависимости параметров и метрик
    param_slice = vis.plot_slice(study)
    param_slice.write_html(f'C:/Dima/Projects/Cuda/LOM/optuna_visualizations/param_slice_{timestamp}.html')

    # Обучение с лучшими гиперпараметрами
    best_params = study.best_params
    model = YOLO('yolov8m.pt')

    model.train(
        data='C:/Dima/Projects/Cuda/LOM/dataset.yaml',
        epochs=best_params['epochs'],
        imgsz=640,
        batch=-1,
        lr0=best_params['lr0'],
        weight_decay=best_params['weight_decay'],
        momentum=best_params['momentum'],
        optimizer=best_params['optimizer'],
        freeze=[0],
        warmup_epochs=2.0,
        cos_lr=True,
        rect=False,
        augment=True,
        patience=200,
        device='cuda'
    )

    # Оценка модели на валидационном наборе
    metrics = model.val()
    print(metrics)

    # Использование модели для предсказаний на новых изображениях
    results = model('C:/Dima/Projects/Cuda/LOM/test_image.jpg')

    # Обработка результатов
    for result in results:
        result.show()  # Показать результаты
        result.save('C:/Dima/Projects/Cuda/LOM/results')  # Сохранить результаты
