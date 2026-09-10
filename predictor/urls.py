from django.urls import path
from . import views

app_name = 'predictor'

urlpatterns = [
    path(
        'dashboard/',
        views.dashboard_view,
        name='dashboard',
    ),
    path(
        'report/',
        views.generate_report_view,
        name='report',
    ),
    path('track/', views.track_record, name='track_record'),
    path('track/delete/<int:pk>/',
         views.delete_workout_log, name='delete_workout_log'),
]

