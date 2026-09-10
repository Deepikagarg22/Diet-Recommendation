from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import login, logout
from django.contrib.auth.views import LoginView
from django.contrib.auth.decorators import login_required
from .forms import (
    UserRegistrationForm, UserLoginForm,
    UserUpdateForm, ProfileUpdateForm,
)


def register_view(request):
    """Handle user registration with profile data."""
    if request.user.is_authenticated:
        return redirect('predictor:dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(
                request,
                f'Welcome {user.first_name}! '
                f'Your account has been created successfully.'
            )
            return redirect('predictor:dashboard')
        else:
            messages.error(
                request,
                'Please correct the errors below.'
            )
    else:
        form = UserRegistrationForm()

    return render(request, 'accounts/register.html', {'form': form})


class CustomLoginView(LoginView):
    """Custom login view with styled form."""
    form_class = UserLoginForm
    template_name = 'accounts/login.html'
    redirect_authenticated_user = True

    def form_valid(self, form):
        messages.success(
            self.request,
            f'Welcome back, {form.get_user().first_name}!'
        )
        return super().form_valid(form)

    def form_invalid(self, form):
        messages.error(
            self.request,
            'Invalid username or password.'
        )
        return super().form_invalid(form)


def logout_view(request):
    """Log the user out."""
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('home')


@login_required
def profile_view(request):
    """Display and update user profile."""
    if request.method == 'POST':
        user_form = UserUpdateForm(
            request.POST, instance=request.user
        )
        profile_form = ProfileUpdateForm(
            request.POST, instance=request.user.profile
        )

        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            messages.success(
                request,
                'Your profile has been updated successfully!'
            )
            return redirect('accounts:profile')
        else:
            messages.error(
                request,
                'Please correct the errors below.'
            )
    else:
        user_form = UserUpdateForm(instance=request.user)
        profile_form = ProfileUpdateForm(
            instance=request.user.profile
        )

    context = {
        'user_form': user_form,
        'profile_form': profile_form,
        'profile': request.user.profile,
    }
    return render(request, 'accounts/profile.html', context)