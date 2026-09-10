from collections import defaultdict
from datetime import datetime, timedelta

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.db.models import Count

from .ml_utils import predictor
from .models import WorkoutLog
from .forms import WorkoutLogForm


@login_required
def dashboard_view(request):
    """Main dashboard — shows profile summary & quick stats."""
    profile = request.user.profile

    context = {
        'profile': profile,
        'is_complete': profile.is_complete,
    }

    if profile.is_complete:
        context['bmi'] = profile.bmi
        context['bmi_category'] = profile.bmi_category

    # ─── Latest 5 logs for dashboard preview ───
    recent_logs = (
        WorkoutLog.objects
        .filter(user=request.user)
        .order_by('-date', '-entry_time')[:5]
    )
    context['recent_logs'] = recent_logs

    return render(request, 'predictor/dashboard.html', context)

@login_required
def generate_report_view(request):
    profile = request.user.profile

    if not profile.is_complete:
        messages.warning(
            request,
            'Please complete your profile before '
            'generating a report.'
        )
        return redirect('accounts:profile')

    try:
        result = predictor.predict(
            age=profile.age,
            gender=profile.gender,
            height_cm=profile.height_cm,
            weight_kg=profile.weight_kg,
            activity_level=profile.activity_level,
            purpose=profile.purpose,
            diet_pref=profile.diet_preference,
        )

        context = {
            'profile': profile,
            'user': request.user,
            'result': result,
        }
        return render(
            request, 'predictor/report.html', context
        )

    except FileNotFoundError as e:
        messages.error(
            request, f'ML Model not found: {str(e)}'
        )
        return redirect('predictor:dashboard')

    except Exception as e:
        messages.error(
            request, f'Prediction error: {str(e)}'
        )
        return redirect('predictor:dashboard')


@login_required
def track_record(request):
    """Show the form and all workout logs for the current user."""

    if request.method == 'POST':
        form = WorkoutLogForm(request.POST)
        if form.is_valid():
            log = form.save(commit=False)
            log.user = request.user
            log.save()
            messages.success(
                request, 'Workout logged successfully!'
            )
            return redirect('predictor:track_record')
    else:
        form = WorkoutLogForm()

    # ─── All logs, sorted: LATEST on top ───
    logs = (
        WorkoutLog.objects
        .filter(user=request.user)
        .order_by('-date', '-entry_time')
    )

    # ── Quick stats ──
    total_workouts = logs.count()

    today = datetime.today().date()
    week_start = today - timedelta(days=today.weekday())
    week_logs = logs.filter(
        date__gte=week_start, date__lte=today
    )
    weekly_count = week_logs.count()

    weekly_minutes = sum(
        l.duration_minutes for l in week_logs
    )
    weekly_hours = weekly_minutes // 60
    weekly_remaining = weekly_minutes % 60

    workout_minutes = defaultdict(int)
    for log_entry in logs:
        workout_minutes[log_entry.workout_type] += log_entry.duration_minutes

    favourite_workout = None
    if workout_minutes:
        top_type = max(workout_minutes, key=workout_minutes.get)
        for code, label in WorkoutLog.WORKOUT_CHOICES:
            if code == top_type:
                favourite_workout = label
                break

    context = {
        'form': form,
        'logs': logs[:30],
        'total_workouts': total_workouts,
        'weekly_count': weekly_count,
        'weekly_hours': weekly_hours,
        'weekly_remaining': weekly_remaining,
        'favourite_workout': favourite_workout,
    }
    return render(
        request, 'predictor/track_record.html', context
    )


# ────────────────────────────────────────────
#  Delete a workout log
# ────────────────────────────────────────────
@login_required
def delete_workout_log(request, pk):
    log = get_object_or_404(
        WorkoutLog, pk=pk, user=request.user
    )
    if request.method == 'POST':
        log.delete()
        messages.success(request, 'Workout log deleted.')
    return redirect('predictor:track_record')