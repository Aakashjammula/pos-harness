import { AuthForm } from "@/components/AuthForm";
import { GuestGuard } from "@/components/GuestGuard";

export default function LoginPage() {
  return (
    <GuestGuard>
      <AuthForm mode="login" />
    </GuestGuard>
  );
}
